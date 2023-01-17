from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
import os

from pylab import *

import seaborn as sns

import pandas as pd

from collections import Counter
sns.set()
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")

df
df.info()
df.columns
plt.figure(figsize=(20, 20))

c = 1

for col in list(df.columns[0:-1]):

    plt.subplot(6, 5, c)

    plt.boxplot(df[col])

    plt.legend([col])

    c+=1

    
sns.distplot(df.Time);
sns.boxenplot(y=df.Amount)
sns.countplot(x="Class", data=df)
counter = Counter(df["Class"])



print(f"""

# NotFraud: {counter[0]}

# Fraud: {counter[1]}



# ratio: {counter[1]/counter[0]}

""")
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
sns.pairplot(data=df, x_vars=["V17", "V20"], y_vars=['Amount']);
sns.pairplot(data=df, x_vars=["V2", "V5"], y_vars=['Amount']);
sns.pairplot(data=df, x_vars=["V3"], y_vars=['Time']);
from sklearn.model_selection import train_test_split
df_X, df_y = df[list(df.columns[:-1])], df["Class"]



X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=3)
# train data ratio of n_frad / n_not_fraud

counter = Counter(y_train)

print(f"train data ratio of n_frad / n_not_fraud: {counter[1]/counter[0]}")
# test data ratio of n_frad / n_not_fraud

counter = Counter(y_test)

print(f"test data: {counter}", end="\n\n")

print(f"test data ratio of n_frad / n_not_fraud: {counter[1]/counter[0]}")
from imblearn.over_sampling import SMOTE
X_samp, y_samp = SMOTE(random_state=1).fit_sample(X_train, y_train)
print(f"""

X_samp.shape: {X_samp.shape}

y_samp.shape: {y_samp.shape}

""")
counter = Counter(y_samp)



print(f"""

* After SMOTE



# NotFraud: {counter[0]}

# Fraud: {counter[1]}



-> Now it's balanced!

""")
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
random_forest_clf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
random_forest_clf.fit(X_samp, y_samp)
import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization

from sklearn.preprocessing import StandardScaler
# input data scaling for training stability



scalerX = StandardScaler()

sc_X_samp = scalerX.fit_transform(X_samp)
tf.keras.backend.clear_session()
input_layer = tf.keras.Input(shape=(30,))



hl = Dense(64, activation="relu", kernel_initializer="he_normal")(input_layer)

hl = BatchNormalization()(hl)



hl = Dense(64, activation="relu", kernel_initializer="he_normal")(hl)

hl = BatchNormalization()(hl)



hl = Dense(64, activation="relu", kernel_initializer="he_normal")(hl)

hl = BatchNormalization()(hl)



hl = Dense(64, activation="relu", kernel_initializer="he_normal")(hl)

hl = BatchNormalization()(hl)



output_layer = Dense(1, activation="sigmoid")(hl)



model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
model.compile(loss="binary_crossentropy", optimizer="adam", 

              metrics=[tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
model.summary()
hist = model.fit(sc_X_samp, y_samp, batch_size=2**14, epochs=20, validation_split=0.1)
# training result plot



plt.plot(hist.history["loss"], label="loss")

plt.legend();
# training result plot



plt.plot(hist.history["precision"], label="precision")

plt.plot(hist.history["recall"], label="recall")

plt.plot(hist.history["val_precision"], label="val_precision")

plt.plot(hist.history["val_recall"], label="val_recall")

plt.legend();
from sklearn.metrics import classification_report, confusion_matrix
# on training dataset



y_pred = random_forest_clf.predict(X_samp)

print(classification_report(y_samp, y_pred))
# on test dataset



y_pred = random_forest_clf.predict(X_test)

print(classification_report(y_test, y_pred))
fi = pd.DataFrame()



fi['features'] = list(X_samp.columns)

fi['values'] = random_forest_clf.feature_importances_



fi = fi.sort_values(by="values", ascending=False);



sns.catplot(x="features", y="values", data=fi, kind="bar", 

            aspect=4, height=4);
# on test dataset



y_pred = model.predict(scalerX.transform(X_test))
print(classification_report(y_test, np.round(y_test)))
confusion_matrix(y_test, np.round(y_test), labels=[0, 1])