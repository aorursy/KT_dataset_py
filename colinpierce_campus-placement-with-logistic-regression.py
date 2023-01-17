import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.info()
df = df.drop('sl_no',axis=1)



df.rename(inplace=True,columns={'gender':'Gender','ssc_p':'Secondary Ed %','ssc_b':'Secondary Board',

                   'hsc_p':'Higher Secondary Ed %','hsc_b':'Higher Secondary Board',

                   'hsc_s':'Higher Secondary Specialization','degree_p':'Degree %',

                   'degree_t':'Under Grad Field','workex':'Work Exp','etest_p':'Employability Test %',

                   'specialisation':'Post Grad Specialization','mba_p':'MBA %','status':'Placement Status',

                                'salary':'Salary'})
sns.countplot(x='Placement Status',data= df)
(len(df[df['Placement Status'] == 'Placed']) / len(df['Placement Status']))*100
sns.countplot(x='Placement Status',data= df,hue='Gender')
def feature_classification_percent(dataframe, feature, goalfeature, positive_goal_value):

    

    unique_lst = []

    unique_dic = {}

    

    # Add all unique values in feature to a list

    for i in range(len(dataframe[feature].unique())):

        unique_lst.append(dataframe[feature].unique()[i])



    

    # Count the amount for each value in feature and count the amount of each value in that feature that the goalfeature is 1

    # Calculate the percentage of that feature that the goalfeature is 1

    # Add information for each value to a calculated values list

    tot_lst = []

    pos_lst = []

    pos_perc_lst = []

    

    for j in range(len(unique_lst)):

        tot = dataframe[dataframe[feature] == unique_lst[j]][goalfeature].count()

        pos = len(dataframe[(dataframe[feature] == unique_lst[j]) & (dataframe[goalfeature] == positive_goal_value)])

        pos_perc = str(round((pos / tot) * 100,1)) + '%'

        

        tot_lst.append(tot)

        pos_lst.append(pos)

        pos_perc_lst.append(pos_perc)

        

        

    # Convert these lists into Series and create index Series

    tot_series = pd.Series(tot_lst)

    pos_series = pd.Series(pos_lst)

    pos_perc_series = pd.Series(pos_perc_lst)

    value_series = pd.Series(unique_lst)

    

    #Create Dataframe from Series

    feature_dataframe = pd.DataFrame({'Values':value_series,'Total Amount':tot_series,'Positive Amount':pos_series,'Positive Percentage':pos_perc_series})

    feature_dataframe.set_index('Values',inplace=True)

    feature_dataframe.sort_values(by='Positive Percentage',inplace=True, ascending=False)

   

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        display(feature_dataframe)
feature_classification_percent(df,'Gender','Placement Status','Placed')
sns.boxplot('Secondary Ed %','Placement Status',data=df)
sns.countplot(x='Placement Status',data= df,hue='Secondary Board')
feature_classification_percent(df,'Secondary Board','Placement Status','Placed')
sns.boxplot('Higher Secondary Ed %','Placement Status',data=df)
sns.countplot(x='Placement Status',data= df,hue='Higher Secondary Board')
feature_classification_percent(df,'Higher Secondary Board','Placement Status','Placed')
sns.countplot(x='Higher Secondary Specialization',data= df,hue='Placement Status')
feature_classification_percent(df,'Higher Secondary Specialization','Placement Status','Placed')
sns.boxplot('Degree %','Placement Status',data=df)
sns.countplot(x='Placement Status',data= df,hue='Under Grad Field')
feature_classification_percent(df,'Under Grad Field','Placement Status','Placed')
sns.countplot(x='Placement Status',data= df,hue='Work Exp')
sns.boxplot('Employability Test %','Placement Status',data=df)
sns.countplot(x='Placement Status',data= df,hue='Post Grad Specialization')
feature_classification_percent(df,'Post Grad Specialization','Placement Status','Placed')
sns.boxplot('MBA %','Placement Status',data=df)
df.drop(['Secondary Board','Higher Secondary Board'],axis=1,inplace=True)
df['Gender'] = df['Gender'].map({'M':0,'F':1})

df['Work Exp'] = df['Work Exp'].map({'No':0,'Yes':1})

df['Placement Status'] = df['Placement Status'].map({'Not Placed':0,'Placed':1})
df = pd.get_dummies(df)
sns.boxplot(df['Salary'])
df = df.drop(df[df['Salary'] > 400000].index)
df.drop('Salary',axis=1,inplace=True)
plt.figure(figsize=(14,8))

sns.heatmap(df.corr(),annot=True)
df = df.drop(['Higher Secondary Specialization_Science','Under Grad Field_Sci&Tech',

                'Post Grad Specialization_Mkt&HR'],axis=1)
plt.figure(figsize=(14,8))

sns.heatmap(df.corr(),annot=True)
df = df.drop(['Gender','Under Grad Field_Comm&Mgmt','Under Grad Field_Others','MBA %',

                           'Higher Secondary Specialization_Arts',

                           'Higher Secondary Specialization_Commerce'],axis=1)
plt.figure(figsize=(14,8))

sns.heatmap(df.corr(),annot=True)
X = df.drop('Placement Status',axis=1)

y = df['Placement Status']
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=42)
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))