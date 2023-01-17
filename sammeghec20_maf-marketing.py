import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import warnings

warnings.filterwarnings('ignore')
import pandas as pd



email_opened = pd.read_csv("../input/email_opened_table.csv")

link_clicked = pd.read_csv("../input/link_clicked_table.csv")

emails = pd.read_csv("../input/email_table.csv")

"emails {}, opened {}, clicked {}, email opened {:.2f}, link clicked {:.2f}".format(

    emails.shape[0],

    email_opened.shape[0],

    link_clicked.shape[0],

    100 * email_opened.shape[0]/emails.shape[0],

    100 * link_clicked.shape[0]/emails.shape[0]

)
click_vals = link_clicked['email_id'].values
def clicks_encoder(x):

    if x in click_vals:

        return 1

    else:

        return 0
emails['click'] = emails['email_id'].apply(lambda x:clicks_encoder(x))
sns.countplot(x='email_text',hue='click',data=emails);
sns.countplot(x='email_version',hue='click',data=emails);
sns.countplot(x='hour',hue='click',data=emails);
sns.countplot(x='weekday',hue='click',data=emails);
sns.countplot(x='user_country',hue='click',data=emails);
sns.countplot(x='user_past_purchases',hue='click',data=emails);
def get_conversion_rate(hue_label):

    dummy = emails.groupby('click')[hue_label].value_counts().unstack().fillna(0)

    dummy = dummy.divide(dummy.iloc[0])*100

    return dummy.iloc[1]
get_conversion_rate("email_text")
get_conversion_rate("email_version")
get_conversion_rate("hour")
get_conversion_rate("weekday")
get_conversion_rate("user_country")
import seaborn as sns

corr = emails_joined.corr()

# plot the heatmap

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
#adding new features to the email table

link_clicked['clicked']=1

email_opened['opened']=1



emails_joined = emails.set_index('email_id').join(

    email_opened.set_index('email_id'), 

    lsuffix='_caller', 

    rsuffix='_other'

).join(

    link_clicked.set_index('email_id'), 

    lsuffix='_caller', 

    rsuffix='_other'

)

emails_joined = emails_joined.fillna(0)

emails_joined.tail()
#this is the helper file

#the helper is attached if MAF team is used to call helpers

from sklearn.preprocessing import LabelEncoder





def encode_label(values):

    encoder = LabelEncoder()

    encoder.fit(values)



    return encoder.transform(values), encoder

#coding features as one can only feed numbers to the neural nets





emails_joined["email_text"], email_text_encoder = encode_label(emails_joined["email_text"].values)



emails_joined["email_version"], email_version_encoder = encode_label(emails_joined["email_version"].values)



emails_joined["weekday"], weekday_encoder = encode_label(emails_joined["weekday"].values)



emails_joined["user_country"], user_country_encoder = encode_label(emails_joined["user_country"].values)

from sklearn.model_selection import train_test_split



x_cols = ["email_text", "email_version", "weekday", "user_country", "hour"]

y_cols = ["opened", "clicked"]



x_data = emails_joined[x_cols]

y_data = emails_joined[y_cols]



X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
#data normalizing method

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

ss.fit(X_train)



X_train_standard = ss.transform(X_train)
from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(12, input_dim=5, activation='relu'))

model.add(Dense(12, activation='relu'))

model.add(Dense(2, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
model.fit(X_train_standard, y_train, epochs=10, batch_size=32)
X_test_standard = ss.transform(X_test)

model.evaluate(X_test_standard, y_test)

#testing on one instance by calling the trained model



def get_label(confidence):

    return "yes" if confidence >= 0.5 else "no"



result = model.predict(X_test_standard[:1])



predicted_clicked = []

actual_clicked = []

predicted_opened = []

actual_opened = []



for y_hat, y_true in zip(result, y_test[:1].values):

    for y_h, y_t, y_col in zip(y_hat, y_true, y_cols):

        if y_col == "clicked":

            predicted_clicked.append(get_label(y_h))

            actual_clicked.append(get_label(y_t))

        else:

            predicted_opened.append(get_label(y_h))

            actual_opened.append(get_label(y_t))



result = pd.DataFrame()

result["actual_clicked"] = actual_clicked

result["actual_opened"] = actual_opened

result["predicted_clicked"] = predicted_clicked

result["predicted_opened"] = predicted_opened

print(result)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train_standard, y_train)

y_pred = rf.predict(X_test_standard)

print("Accuracy Metric")

print(accuracy_score(y_test, y_pred))
