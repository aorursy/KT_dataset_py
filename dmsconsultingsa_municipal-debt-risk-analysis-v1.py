from __future__ import absolute_import, division, print_function, unicode_literals



import pathlib



import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier



import os

import tempfile



print(tf.__version__)
dataset_path = '../input/municipal-debt-risk-analysis/MunicipalDebtAnalysis.csv'

column_names = ['AccCategoryID','AccCategory','AccCategoryAbbr','PropertyValue', 'PropertySize', 'TotalBilling','AverageBilling','TotalReceipting','AverageReceipting','TotalDebt', 

                'TotalWriteOff','CollectionRatio','DebtBillingRatio','TotalElectricityBill','HasIDNo','BadDebtIndic']

df = pd.read_csv(dataset_path, skiprows=1,names=column_names)

df.head()
ax = sns.countplot(x=df.BadDebtIndic)

plt.title("Count by Bad Debt Indicator");
BadDebtPerc = len(df[df.BadDebtIndic==1])/len(df)*100

print('Total Records: ' + f'{len(df):,d}')

print('Bad Debt: ' + f'{len(df[df.BadDebtIndic==1]):,d}' + ' (' + f'{BadDebtPerc:,.2f}'+'%)')
df_stats = df.describe()

df_stats = df_stats.transpose()

df_stats
drop1=df.index[df["TotalBilling"] <= 0].tolist()

drop2=df.index[df["TotalReceipting"] < 0].tolist()

droptotal=drop1+drop2

dfclean=df.drop(df.index[droptotal])



df_stats = dfclean.describe()

df_stats = df_stats.transpose()

df_stats
DroppedRecs = len(df)-len(dfclean)

DroppedPerc = DroppedRecs/len(df)*100

print('Total Records Dropped: ' + f'{DroppedRecs:,d}' + ' (' + f'{DroppedPerc:,.2f}'+'%)')
df = dfclean
df['CollectionRatio'] = np.where((df.CollectionRatio >1),1,df.CollectionRatio)
sns.boxplot(x='BadDebtIndic',y='CollectionRatio', data=df);
df.insert(14, 'HasElectricity', 0)

df['HasElectricity'] = np.where(df['TotalElectricityBill']!= 0, 1, 0)

df = df.drop(['TotalElectricityBill'], axis = 1);
df = df.drop(['TotalDebt','TotalWriteOff','DebtBillingRatio'], axis = 1);
df_stats = df.describe()

df_stats = df_stats.transpose()

df_stats
df['PropertyValueLog'] = np.log(df[['PropertyValue']].replace(0, 0.01))

fig, axes = plt.subplots(1,2)

plot1 = df.hist('PropertyValue', ax=axes[0])

plot2 = df.hist('PropertyValueLog', ax=axes[1])
df['PropertySizeLog'] = np.log(df[['PropertySize']].replace(0, 0.01))

fig, axes = plt.subplots(1,2)

plot1 = df.hist('PropertySize', ax=axes[0])

plot2 = df.hist('PropertySizeLog', ax=axes[1])
df['TotalBillingLog'] = np.log(df[['TotalBilling']].replace(0, 0.01))

fig, axes = plt.subplots(1,2)

plot1 = df.hist('TotalBilling', ax=axes[0])

plot2 = df.hist('TotalBillingLog', ax=axes[1])
df['AverageBillingLog'] = np.log(df[['AverageBilling']].replace(0, 0.01))

fig, axes = plt.subplots(1,2)

plot1 = df.hist('AverageBilling', ax=axes[0])

plot2 = df.hist('AverageBillingLog', ax=axes[1])
df['TotalReceiptingLog'] = np.log(df[['TotalReceipting']].replace(0, 0.01))

fig, axes = plt.subplots(1,2)

plot1 = df.hist('TotalReceipting', ax=axes[0])

plot2 = df.hist('TotalReceiptingLog', ax=axes[1])
df['AverageReceiptingLog'] = np.log(df[['AverageReceipting']].replace(0, 0.01))

fig, axes = plt.subplots(1,2)

plot1 = df.hist('AverageReceipting', ax=axes[0])

plot2 = df.hist('AverageReceiptingLog', ax=axes[1])
tmpCol = df['BadDebtIndic']

df.drop(labels=['BadDebtIndic'], axis=1,inplace = True)

df.insert(18, 'BadDebtIndic', tmpCol)
plt.figure(figsize=(30,10))

cor = df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
cor_target = abs(cor["BadDebtIndic"])

relevant_features = cor_target[cor_target>0.5]

relevant_features
plt.figure(figsize=(20, 6))

plt.subplot(1,2,1)

sns.distplot(df[df.BadDebtIndic==0].CollectionRatio,kde=True,color="Blue", label="Collection Ratio for non-Bad Debt")

sns.distplot(df[df.BadDebtIndic==1].CollectionRatio,kde=True,color = "Gold", label = "Collection Ratio for Bad Debt")

plt.title("Histograms for Collection Ratio by Bad Debt")

plt.legend()

plt.subplot(1,2,2)

sns.boxplot(x=df.BadDebtIndic,y=df.CollectionRatio)

plt.title("Boxplot for Collection Ratio by Bad Debt");
plt.figure(figsize=(20, 6))

plt.subplot(1,2,1)

sns.distplot(df[df.BadDebtIndic==0].HasIDNo,kde=False,color="Blue", label="Has ID No for non-Bad Debt")

sns.distplot(df[df.BadDebtIndic==1].HasIDNo,kde=False,color = "Gold", label = "Has ID No for Bad Debt")

plt.title("Histograms for Has ID No by Bad Debt")

plt.legend()

plt.subplot(1,2,2)

sns.boxplot(x=df.BadDebtIndic,y=df.HasIDNo)

plt.title("Boxplot for Has ID No by Bad Debt");
X = df.iloc[:,4:18]

y = df.iloc[:,-1]

model = RandomForestClassifier()

model.fit(X,y)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show();
df.PropertyValueLog.quantile([0.25,0.5,0.75])

PropValQ1 = df.PropertyValueLog.quantile([0.25])

PropValQ2 = df.PropertyValueLog.quantile([0.5])

PropValQ3 = df.PropertyValueLog.quantile([0.75])



plt.figure(figsize=(20, 10))

plt.subplot(2,4,1)

sns.distplot(df[df.PropertyValueLog<=PropValQ1[0.25]].BadDebtIndic,kde=False,color="Green")

plt.title("Property Values for Q1");

plt.subplot(2,4,2)

sns.distplot(df[(df["PropertyValueLog"] > PropValQ1[0.25]) & (df["PropertyValueLog"] <= PropValQ2[0.5])].BadDebtIndic,kde=False,color="Blue")

plt.title("Property Values for Q2");

plt.subplot(2,4,3)

sns.distplot(df[(df["PropertyValueLog"] > PropValQ2[0.5]) & (df["PropertyValueLog"] <= PropValQ3[0.75])].BadDebtIndic,kde=False,color="Orange")

plt.title("Property Values for Q3");

plt.subplot(2,4,4)

sns.distplot(df[df.PropertyValueLog>PropValQ3[0.75]].BadDebtIndic,kde=False,color="Red")

plt.title("Property Values for Q4");



plt.subplot(2,4,5)

df.PropertyValueLog.groupby(df[df.PropertyValueLog<=PropValQ1[0.25]].BadDebtIndic).count().plot(kind='pie')

plt.axis('equal')

plt.subplot(2,4,6)

df.PropertyValueLog.groupby(df[(df["PropertyValueLog"] > PropValQ1[0.25]) & (df["PropertyValueLog"] <= PropValQ2[0.5])].BadDebtIndic).count().plot(kind='pie')

plt.axis('equal')

plt.subplot(2,4,7)

df.PropertyValueLog.groupby(df[(df["PropertyValueLog"] > PropValQ2[0.5]) & (df["PropertyValueLog"] <= PropValQ3[0.75])].BadDebtIndic).count().plot(kind='pie')

plt.axis('equal')

plt.subplot(2,4,8)

df.PropertyValueLog.groupby(df[df.PropertyValueLog>PropValQ3[0.75]].BadDebtIndic).count().plot(kind='pie')

plt.axis('equal');
df.PropertySizeLog.quantile([0.25,0.5,0.75])

PropSizeQ1 = df.PropertySizeLog.quantile([0.25])

PropSizeQ2 = df.PropertySizeLog.quantile([0.5])

PropSizeQ3 = df.PropertySizeLog.quantile([0.75])



plt.figure(figsize=(20, 10))

plt.subplot(2,4,1)

sns.distplot(df[df.PropertySizeLog<=PropSizeQ1[0.25]].BadDebtIndic,kde=False,color="Green")

plt.title("Property Sizes for Q1");

plt.subplot(2,4,2)

sns.distplot(df[(df["PropertySizeLog"] > PropSizeQ1[0.25]) & (df["PropertySizeLog"] <= PropSizeQ2[0.5])].BadDebtIndic,kde=False,color="Blue")

plt.title("Property Sizes for Q2");

plt.subplot(2,4,3)

sns.distplot(df[(df["PropertySizeLog"] > PropSizeQ2[0.5]) & (df["PropertySizeLog"] <= PropSizeQ3[0.75])].BadDebtIndic,kde=False,color="Orange")

plt.title("Property Sizes for Q3");

plt.subplot(2,4,4)

sns.distplot(df[df.PropertySizeLog>PropSizeQ3[0.75]].BadDebtIndic,kde=False,color="Red")

plt.title("Property Sizes for Q4");



plt.subplot(2,4,5)

df.PropertySizeLog.groupby(df[df.PropertySizeLog<=PropSizeQ1[0.25]].BadDebtIndic).count().plot(kind='pie')

plt.axis('equal')

plt.subplot(2,4,6)

df.PropertySizeLog.groupby(df[(df["PropertySizeLog"] > PropSizeQ1[0.25]) & (df["PropertySizeLog"] <= PropSizeQ2[0.5])].BadDebtIndic).count().plot(kind='pie')

plt.axis('equal')

plt.subplot(2,4,7)

df.PropertySizeLog.groupby(df[(df["PropertySizeLog"] > PropSizeQ2[0.5]) & (df["PropertySizeLog"] <= PropSizeQ3[0.75])].BadDebtIndic).count().plot(kind='pie')

plt.axis('equal')

plt.subplot(2,4,8)

df.PropertySizeLog.groupby(df[df.PropertySizeLog>PropSizeQ3[0.75]].BadDebtIndic).count().plot(kind='pie')
df = pd.get_dummies(df,prefix=['AccCat_'], columns = ['AccCategoryAbbr'], drop_first=False)

df = df.drop(['AccCategoryID','AccCategory','PropertyValue','PropertySize','TotalBilling','AverageBilling','TotalReceipting','AverageReceipting','HasElectricity','TotalBillingLog','AverageBillingLog'], axis = 1);
for col in df.columns: 

    print(col) 
train_df, test_df = train_test_split(df, test_size=0.2)

train_df, val_df = train_test_split(train_df, test_size=0.2)
plt.figure(figsize=(20, 5))

plt.subplot(1,3,1)

ax = sns.countplot(x=train_df.BadDebtIndic)

plt.title("Train Set");

plt.subplot(1,3,2)

ax = sns.countplot(x=test_df.BadDebtIndic)

plt.title("Test Set");

plt.subplot(1,3,3)

ax = sns.countplot(x=val_df.BadDebtIndic)

plt.title("Validation Set");
train_labels = np.array(train_df.pop('BadDebtIndic'))

bool_train_labels = train_labels != 0

val_labels = np.array(val_df.pop('BadDebtIndic'))

test_labels = np.array(test_df.pop('BadDebtIndic'))



train_features = np.array(train_df)

val_features = np.array(val_df)

test_features = np.array(test_df)
scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)



val_features = scaler.transform(val_features)

test_features = scaler.transform(test_features)



print('Training labels shape:', train_labels.shape)

print('Validation labels shape:', val_labels.shape)

print('Test labels shape:', test_labels.shape)



print('Training features shape:', train_features.shape)

print('Validation features shape:', val_features.shape)

print('Test features shape:', test_features.shape)
METRICS = [

      keras.metrics.TruePositives(name='tp'),

      keras.metrics.FalsePositives(name='fp'),

      keras.metrics.TrueNegatives(name='tn'),

      keras.metrics.FalseNegatives(name='fn'), 

      keras.metrics.BinaryAccuracy(name='accuracy'),

      keras.metrics.Precision(name='precision'),

      keras.metrics.Recall(name='recall'),

      keras.metrics.AUC(name='auc'),

]



def make_model(metrics = METRICS, output_bias=None):

  if output_bias is not None:

    output_bias = tf.keras.initializers.Constant(output_bias)

  model = keras.Sequential([

      keras.layers.Dense(

          16, activation='relu',

          input_shape=(train_features.shape[-1],)),

      keras.layers.Dropout(0.5),

      keras.layers.Dense(1, activation='sigmoid',

                         bias_initializer=output_bias),

  ])



  model.compile(

      optimizer=keras.optimizers.Adam(lr=1e-3),

      loss=keras.losses.BinaryCrossentropy(),

      metrics=metrics)



  return model
EPOCHS = 100

BATCH_SIZE = 32



early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_auc', 

    verbose=1,

    patience=10,

    mode='max',

    restore_best_weights=True)
model = make_model()

model.summary()
model.predict(train_features[:10])
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)

print("Loss: {:0.4f}".format(results[0]))
neg, pos = np.bincount(df['BadDebtIndic'])

initial_bias = np.log([pos/neg])

initial_bias
model = make_model(output_bias = initial_bias)

model.predict(train_features[:10])
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)

print("Loss: {:0.4f}".format(results[0]))
initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')

model.save_weights(initial_weights)
model = make_model()

model.load_weights(initial_weights)

model.layers[-1].bias.assign([0.0])

zero_bias_history = model.fit(

    train_features,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=20,

    validation_data=(val_features, val_labels), 

    verbose=0)
model = make_model()

model.load_weights(initial_weights)

careful_bias_history = model.fit(

    train_features,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=20,

    validation_data=(val_features, val_labels), 

    verbose=0)
def plot_loss(history, label, n):

  plt.semilogy(history.epoch,  history.history['loss'],

               color=colors[n], label='Train '+label)

  plt.semilogy(history.epoch,  history.history['val_loss'],

          color=colors[n], label='Val '+label,

          linestyle="--")

  plt.xlabel('Epoch')

  plt.ylabel('Loss')

  

  plt.legend()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plot_loss(zero_bias_history, "Zero Bias", 0)

plot_loss(careful_bias_history, "Careful Bias", 1)
model = make_model()

model.load_weights(initial_weights)

baseline_history = model.fit(

    train_features,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks = [early_stopping],

    validation_data=(val_features, val_labels))
def plot_metrics(history):

  metrics =  ['loss', 'auc', 'precision', 'recall']

  for n, metric in enumerate(metrics):

    name = metric.replace("_"," ").capitalize()

    plt.subplot(2,2,n+1)

    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')

    plt.plot(history.epoch, history.history['val_'+metric],

             color=colors[0], linestyle="--", label='Val')

    plt.xlabel('Epoch')

    plt.ylabel(name)

    if metric == 'loss':

      plt.ylim([0, plt.ylim()[1]])

    elif metric == 'auc':

      plt.ylim([0.8,1])

    else:

      plt.ylim([0,1])



    plt.legend()
mpl.rcParams['figure.figsize'] = (12, 10)

plot_metrics(baseline_history)
train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)

test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
def plot_cm(labels, predictions, p=0.5):

  cm = confusion_matrix(labels, predictions > p)

  plt.figure(figsize=(5,5))

  sns.heatmap(cm, annot=True, fmt="d")

  plt.title('Confusion matrix @{:.2f}'.format(p))

  plt.ylabel('Actual label')

  plt.xlabel('Predicted label')



  print('Non-Bad Debts Detected (True Negatives): ', cm[0][0])

  print('Non-Bad Debts Incorrectly Detected (False Positives): ', cm[0][1])

  print('Bad Debts Missed (False Negatives): ', cm[1][0])

  print('Bad Debts Detected (True Positives): ', cm[1][1])

  print('Total Bad Debts: ', np.sum(cm[1]))
baseline_results = model.evaluate(test_features, test_labels,

                                  batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(model.metrics_names, baseline_results):

  print(name, ': ', value)

print()



plot_cm(test_labels, test_predictions_baseline)
def plot_roc(name, labels, predictions, **kwargs):

  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)



  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)

  plt.xlabel('False positives [%]')

  plt.ylabel('True positives [%]')

  plt.grid(True)

  ax = plt.gca()

  ax.set_aspect('equal')
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])

plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plt.legend(loc='lower right')