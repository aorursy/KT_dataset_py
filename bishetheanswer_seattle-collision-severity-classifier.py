import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
import itertools
%matplotlib inline 
#Create a directory to save plots
! mkdir plots
# Set the seed
seed = 27912
np.random.seed(seed)
#Read data set
df = pd.read_csv("../input/seattlecollisions/Data-Collisions.csv")
df.head()
df.shape
missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")   
df['ADDRTYPE'].value_counts()
df['ADDRTYPE'].value_counts().idxmax()
#replace the missing 'ADDRTYPE' values by the most frequent 
df['ADDRTYPE'].replace(np.nan, 'Block', inplace=True)
df['ADDRTYPE'].value_counts()
#replace the missing 'COLLISIONTYPE' values by the most frequent
most_freq = df['COLLISIONTYPE'].value_counts().idxmax()
df['COLLISIONTYPE'].replace(np.nan, most_freq, inplace=True)
#replace the missing 'JUNCTIONTYPE' values by the most frequent
most_freq = df['JUNCTIONTYPE'].value_counts().idxmax()
df['JUNCTIONTYPE'].replace(np.nan, most_freq, inplace=True)
#replace the missing 'UNDERINFL' values by the most frequent
most_freq = df['UNDERINFL'].value_counts().idxmax()
df['UNDERINFL'].replace(np.nan, most_freq, inplace=True)
#replace the missing 'WEATHER' values by the most frequent
most_freq = df['WEATHER'].value_counts().idxmax()
df['WEATHER'].replace(np.nan, most_freq, inplace=True)
#replace the missing 'ROADCOND' values by the most frequent
most_freq = df['ROADCOND'].value_counts().idxmax()
df['ROADCOND'].replace(np.nan, most_freq, inplace=True)
#replace the missing 'LIGHTCOND' values by the most frequent
most_freq = df['LIGHTCOND'].value_counts().idxmax()
df['LIGHTCOND'].replace(np.nan, most_freq, inplace=True)
#drop the columns listed above
df.drop(['INTKEY','EXCEPTRSNCODE','EXCEPTRSNDESC','INATTENTIONIND','PEDROWNOTGRNT','SPEEDING','SDOTCOLNUM','LOCATION'], axis=1, inplace=True)
df.columns
df.shape
df['SEVERITYCODE'].equals(df['SEVERITYCODE.1'])
to_drop = ['SEVERITYCODE.1']
to_study = ['OBJECTID', 'INCKEY', 'COLDETKEY', 'REPORTNO', 'SDOT_COLCODE', 'ST_COLCODE', 'SEGLANEKEY', 'CROSSWALKKEY']

for column in to_study:
    print(column + ": {}".format(len(df[column].unique())))
to_drop.extend(['OBJECTID', 'INCKEY', 'COLDETKEY', 'REPORTNO'])
df[['SEVERITYCODE','SEVERITYDESC']].value_counts()
df[['SDOT_COLCODE','SDOT_COLDESC']].value_counts(sort=False)
df[['ST_COLCODE','ST_COLDESC']].value_counts(sort=False)
to_drop.extend(['SEVERITYDESC','SDOT_COLDESC','ST_COLDESC'])
df[['INCDATE','INCDTTM']].sample(5)
to_drop.extend(['INCDATE'])
df.drop(to_drop, axis=1, inplace=True)
df.columns
df.dtypes
df[['SEVERITYCODE']].value_counts()
df[['STATUS']].value_counts()
df[['ADDRTYPE']].value_counts()
df[['COLLISIONTYPE']].value_counts()
df[['JUNCTIONTYPE']].value_counts()
#replace the missing 'JUNCTIONTYPE' values by the most frequent
most_freq = df['JUNCTIONTYPE'].value_counts().idxmax()
df['JUNCTIONTYPE'].replace('Unknown', most_freq, inplace=True)
df[['UNDERINFL']].value_counts()
df.loc[df.UNDERINFL == '0', 'UNDERINFL'] = "N"
df.loc[df.UNDERINFL == '1', 'UNDERINFL'] = "Y"
df[['UNDERINFL']].value_counts()
df[['WEATHER']].value_counts()
#replace the missing 'WEATHER' values by the most frequent
most_freq = df['WEATHER'].value_counts().idxmax()
df['WEATHER'].replace('Unknown', most_freq, inplace=True)
df[['ROADCOND']].value_counts()
#replace the missing 'ROADCOND' values by the most frequent
most_freq = df['ROADCOND'].value_counts().idxmax()
df['ROADCOND'].replace('Unknown', most_freq, inplace=True)
df[['LIGHTCOND']].value_counts()
#replace the missing 'LIGHTCOND' values by the most frequent
most_freq = df['LIGHTCOND'].value_counts().idxmax()
df['LIGHTCOND'].replace('Unknown', most_freq, inplace=True)
df[['HITPARKEDCAR']].value_counts()
to_cat = ['SEVERITYCODE', 'STATUS', 'ADDRTYPE', 'COLLISIONTYPE', 'JUNCTIONTYPE', 'UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND', 'HITPARKEDCAR']
df[to_cat] = df[to_cat].astype('category') 
df[['SDOT_COLCODE']] = df[['SDOT_COLCODE']].astype('object')
df[['INCDTTM']] = df[['INCDTTM']].astype('datetime64')
df.dtypes
df.columns[df.isna().any()].tolist()
df.drop(['ST_COLCODE'], axis=1, inplace=True)
# Split the dataset in training and test
(df, df_test) = train_test_split(
    df,
    train_size=0.7, shuffle=True, random_state=seed)
#sns.set_theme(style='darkgrid')
plt.figure(figsize=(8,6))
sns.countplot(x ='SEVERITYCODE', palette='Set2', data = df) 
plt.title('Severity Code Count', fontsize=18)
plt.xlabel('Severity Code', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.savefig('./plots/1.png')
df["SEVERITYCODE"].value_counts(normalize=True)*100
fig, ax = plt.subplots(figsize=(20,5))
sns.countplot(df['INCDTTM'].dt.year, palette='husl', ax=ax)
ax.set_xlabel('Year', fontsize=18)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_ylabel('Collision Count', fontsize=18)
plt.title('Collision count through years', fontsize=18)
plt.savefig('./plots/2.png')
df['HOUR'] = df['INCDTTM'].dt.hour
df[['HOUR', 'INCDTTM']].sample(5)
bins = [0,4,8,12,16,20,24]
labels = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']
df['TIME'] = pd.cut(df['HOUR'], bins=bins, labels=labels, include_lowest=True)
df[['TIME','HOUR']].sample(5)
df.drop(['HOUR'], axis=1, inplace=True)
def time_of_day_plot(df, title):
    '''
    Creates a countplot visualizing the data throughout the day 
    including the frequency.
    
        Parameters:
            df(DataFrame): Data to be visualized
            title(str): Title for the plot
    '''
    ncount = len(df['TIME'])
    plt.figure(figsize=(12,8))
    ax = sns.countplot(x='TIME', palette='Oranges', data=df)
    plt.title(title, fontsize=18)
    plt.xlabel('Time of the day', fontsize=18)

    # Make twin axis
    ax2=ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax.set_ylabel('Count', fontsize=18)
    ax2.set_ylabel('Frequency [%]', fontsize=18)
    ax.tick_params(axis="x", labelsize=15)

    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom', fontsize=15) # set the alignment of the text

    ax2.set_ylim(0,100*ax.get_ylim()[1]/ncount)

    # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    ax2.grid(None)
time_of_day_plot(df, 'Distribution of Collisions throughout the day')
plt.savefig('./plots/3.png')
plt.figure(figsize=(8,6))
sns.countplot(x ='UNDERINFL', palette='Set2', data = df) 
plt.title('Under Influence Count', fontsize=18)
plt.xlabel('Under Influence', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.savefig('./plots/4.png')
#Filter the dataset with only the rows where UNDERINFL is Y
influenced = df['UNDERINFL'] == 'Y'
influenced = df[influenced]
time_of_day_plot(influenced, 'Distribution of Collisions influenced by alcohol or drugs')
plt.savefig('./plots/5.png')
plt.figure(figsize=(8,6))
sns.countplot(x ='HITPARKEDCAR', palette='Set2', data = df) 
plt.title('Hit Parked Car Count', fontsize=18)
plt.xlabel('Hit Parked Car', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.savefig('./plots/6.png')
#Filter the dataset with only the rows where HITPARKEDCAR is Y
hit = df['HITPARKEDCAR'] == 'Y'
hit = df[hit]
time_of_day_plot(hit, 'Distribution of Collisions when a parked car is hit')
plt.savefig('./plots/7.png')
sns.set_palette(sns.color_palette('magma_r'))
tempdf = hit[(hit['UNDERINFL']=='Y')|(hit['UNDERINFL']=='N')]
fig, ax = plt.subplots(figsize=(7,7))
ax.pie(tempdf['UNDERINFL'].value_counts(), textprops={'color':'white', 'fontsize': 14}, autopct='%1.0f%%', explode=[0,0.1])

lgd = ax.legend(tempdf['UNDERINFL'].unique(),
          title='Under Alcohol/Drugs Influence',
          loc='upper center',
          bbox_to_anchor=(1, 0, 0.5, 1))
#plt.savefig('./plots/8.png')
plt.savefig('./plots/8.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
#Filter the dataset with only the rows where HITPARKEDCAR is Y and UNDERINFL is Y
hit_infl = (df['HITPARKEDCAR'] == 'Y') & (df['UNDERINFL'] == 'Y')
hit_infl = df[hit_infl]
time_of_day_plot(hit_infl, 'Distribution of Collisions when influenced by alcohol/drugs and a parked car is hit')
plt.savefig('./plots/9.png')
plt.figure(figsize=(12,8))
sns.countplot(y ='COLLISIONTYPE', palette='husl', data = df) 
plt.savefig('./plots/10.png')
plt.figure(figsize=(12,8))
sns.countplot(y ='JUNCTIONTYPE', palette='husl', data = df)
plt.savefig('./plots/11.png')
fig, axs = plt.subplots(nrows=3, figsize=(15,15))
sns.countplot(y ='WEATHER', palette='husl', data = df, ax=axs[0]) 
sns.countplot(y ='ROADCOND', palette='husl', data = df, ax=axs[1]) 
sns.countplot(y ='LIGHTCOND', palette='husl', data = df, ax=axs[2])
plt.savefig('./plots/12.png')
from folium import plugins

# simply drop whole row with NaN in "price" column
df_map = df.dropna(subset=['X','Y'], axis=0)

# reset index, because we droped two rows
df_map.reset_index(drop=True, inplace=True)
 
latitude = df_map['Y'].mean()
longitude = df_map['X'].mean()

# let's start with a clean copy of the map of Seattle
seattle_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# instantiate a mark cluster object for the collisions in the dataframe
collisions = plugins.MarkerCluster().add_to(seattle_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng in zip(df_map.Y, df_map.X):
    folium.Marker(
        location=[lat, lng],
        icon=None
    ).add_to(collisions)

# display map
seattle_map
df.drop(['X','Y','INCDTTM'], axis=1, inplace=True)
df.shape
(X, y) = (df.drop('SEVERITYCODE', axis=1), df['SEVERITYCODE'])
to_encode = ['STATUS', 
             'ADDRTYPE', 
             'COLLISIONTYPE',
             'JUNCTIONTYPE',
             'UNDERINFL',
             'WEATHER',
             'ROADCOND',
             'LIGHTCOND',
             'HITPARKEDCAR',
             'TIME']

le = LabelEncoder()

for feat in to_encode:
    X[feat] = le.fit_transform(X[feat].astype(str))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth = 4)
tree_model.fit(X_train, y_train)
predTree = tree_model.predict(X_val)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_val, predTree))
fn=['STATUS','ADDRTYPE','COLLISIONTYPE','PERSONCOUNT','PEDCOUNT','PEDCYLCOUNT','VEHCOUNT','JUNCTIONTYPE','SDOT_COLDOE',
   'UNDERINFL','WEATHER','ROADCOND','LIGHTCOND','SEGLANEKEY','CROSSWALKKEY','HITPARKEDCAR','TIME']
cn=['prop damage', 'injury']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (40,20), dpi=300)

out = tree.plot_tree(tree_model,
               feature_names = fn, 
               class_names=cn,
               fontsize=15,
               filled = True);

for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('black')
        arrow.set_linewidth(3)

plt.savefig('./plots/13.png')
Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_val)
    mean_acc[n-1] = metrics.accuracy_score(y_val, yhat)

    
    std_acc[n-1]=np.std(yhat==y_val)/np.sqrt(yhat.shape[0])

mean_acc
plt.figure(figsize=(10,6))
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.40, color='aquamarine')
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.savefig('./plots/14.png')
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
kNN_model = KNeighborsClassifier(n_neighbors = 16)
LR_model = LogisticRegression(C=0.01, solver='sag', max_iter=1000)
LR_model.fit(X_train, y_train)
predLR = LR_model.predict(X_val)
print("Logistic Regression's Accuracy: ", metrics.accuracy_score(y_val,predLR))
df_test['HOUR'] = df_test['INCDTTM'].dt.hour
bins = [0,4,8,12,16,20,24]
labels = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']
df_test['TIME'] = pd.cut(df_test['HOUR'], bins=bins, labels=labels, include_lowest=True)
df_test.drop(['HOUR'], axis=1, inplace=True)

df_test.drop(['X','Y','INCDTTM'], axis=1, inplace=True)

(X_test, y_test) = (df_test.drop('SEVERITYCODE', axis=1), df_test['SEVERITYCODE'])

to_encode = ['STATUS', 
             'ADDRTYPE', 
             'COLLISIONTYPE',
             'JUNCTIONTYPE',
             'UNDERINFL',
             'WEATHER',
             'ROADCOND',
             'LIGHTCOND',
             'HITPARKEDCAR',
             'TIME']

le = LabelEncoder()

for feat in to_encode:
    X_test[feat] = le.fit_transform(X_test[feat].astype(str))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
#Train the model on the training data
tree_model.fit(X, y)

#Make predictions on the test data
yhat_tree = tree_model.predict(X_test)

#Compute the different metrics
jaccard_tree = metrics.jaccard_score(y_test, yhat_tree)
f1_tree = metrics.f1_score(y_test, yhat_tree, average='weighted') 
acc_tree = metrics.accuracy_score(y_test, yhat_tree)

#Print the results
print("Tree model Accuracy Score", acc_tree)
print("Tree model Jaccard Score: ", jaccard_tree)
print("Tree model F1 Score: ", f1_tree)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat_tree, labels=[1,2])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat_tree))

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Prop damage(1)','Injury(2)'],normalize= False,  title='Confusion matrix')
plt.savefig('./plots/15.png')
#Train the model on the training data
kNN_model.fit(X, y)

#Make predictions on the test data
yhat_kNN = kNN_model.predict(X_test)

#Compute the different metrics
jaccard_kNN = metrics.jaccard_score(y_test, yhat_kNN)
f1_kNN = metrics.f1_score(y_test, yhat_kNN, average='weighted') 
acc_kNN = metrics.accuracy_score(y_test, yhat_kNN)

#Print the results
print("K Nearest Neighbors model Accuracy Score", acc_kNN)
print("K Nearest Neighbors model Jaccard Score: ", jaccard_kNN)
print("K Nearest Neighbors model F1 Score: ", f1_kNN)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat_kNN, labels=[1,2])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat_tree))

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Prop damage(1)','Injury(2)'],normalize= False,  title='Confusion matrix')
plt.savefig('./plots/16.png')
#Train the model on the training data
LR_model.fit(X, y)

#Make predictions on the test data
yhat_LR = LR_model.predict(X_test)
yhat_LR_prob = LR_model.predict_proba(X_test)

#Compute the different metrics
acc_LR = metrics.accuracy_score(y_test, yhat_LR)
jaccard_LR = metrics.jaccard_score(y_test, yhat_LR)
f1_LR = metrics.f1_score(y_test, yhat_LR, average='weighted') 
loss_LR = metrics.log_loss(y_test, yhat_LR_prob)

#Print the results
print("Logistic Regression model Accuracy Score", acc_LR)
print("Logistic Regression model Jaccard Score: ", jaccard_LR)
print("Logistic Regression model F1 Score: ", f1_LR)
print("Logistic Regression mode Log loss ", loss_LR)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat_LR, labels=[1,2])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat_tree))

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Prop damage(1)','Injury(2)'],normalize= False,  title='Confusion matrix')
plt.savefig('./plots/17.png')
#Create lists with values
algorithms = ['Decision Tree', 'K Nearest Neighbors', 'Logistic Regression']
acc_total = [acc_tree, acc_kNN, acc_LR]
jaccard_total = [jaccard_tree, jaccard_kNN, jaccard_LR]
f1_total = [f1_tree, f1_kNN, f1_LR]
loss_total = ['','',loss_LR]

#Create the dictionary
d = {'Algorithm':algorithms, 'Accuracy':acc_total, 'Jaccard':jaccard_total, 'F1-score': f1_total, 'LogLoss': loss_total}

#Create and visualize the DataFrame
results = pd.DataFrame(d)
results.set_index('Algorithm', inplace=True)
results