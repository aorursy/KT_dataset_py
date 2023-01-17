import pandas as pd

import numpy as np
data = pd.read_csv('train.csv')
data
data.describe()
import matplotlib.pyplot as plt

import seaborn as sns
# Compute the correlation matrix

corr = data.corr(method="kendall")



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)



plt.show()
data1 = data.drop(['ID','Class'],axis=1)
data1.describe()
data.columns
df = pd.get_dummies(data1,columns=['Worker Class','Schooling','Enrolled','Married_Life','MIC','MOC', 'Cast', 'Hispanic', 'Sex',

       'MLU', 'Reason','Full/Part','Tax Status','Area', 'State', 'Detailed', 'Summary', 'MSA', 'REG', 'MOVE','Live', 'PREV',

       'Teen', 'COB FATHER', 'COB MOTHER', 'COB SELF','Citizen','Fill'])
y = data['Class']

X = df
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(max_depth=17,random_state=42)

rf.fit(X_train,y_train)

pred=rf.predict(X_test)

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,pred)
rf.fit(X,y)
test_data = pd.read_csv('test_1.csv')
test_df = test_data.drop(['ID'],axis=1)
test_df = pd.get_dummies(test_data,columns=['Worker Class','Schooling','Enrolled','Married_Life','MIC','MOC', 'Cast', 'Hispanic', 'Sex',

       'MLU', 'Reason','Full/Part','Tax Status','Area', 'State', 'Detailed', 'Summary', 'MSA', 'REG', 'MOVE','Live', 'PREV',

       'Teen', 'COB FATHER', 'COB MOTHER', 'COB SELF','Citizen','Fill'])
y_test_pred = rf.predict(test_df)
sub = pd.DataFrame()

sub['ID'] = test_data['ID']

sub['Class'] = y_test_pred

sub.to_csv('submission.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="solution" href="data:text/csv;base64,{payload}" target="_blank">Download the Submission File</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub)