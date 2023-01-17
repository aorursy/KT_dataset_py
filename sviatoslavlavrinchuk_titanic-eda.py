import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import GridSearchCV 

from sklearn.impute import KNNImputer

import category_encoders as ce

from sklearn.model_selection import cross_val_score

pd.set_option('display.max_rows', 1000)
df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")

dfs = [df_train, df_test]

df = df_train.copy() # just to fill free to mess up the data a bit:)
df.head(10)
fig = px.scatter_3d(df, x='Pclass', y='Age', z='Fare', color_continuous_scale=px.colors.sequential.Bluered,

              color='Survived', size ="SibSp", symbol='Sex', size_max=30 , opacity=0.7)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), coloraxis_showscale=False)

fig.show()
df.describe()
df.dtypes
for col in df.columns:

    print(f"{col}: {len(df[col].unique())}")
print(f"Null values if train set: \n\n{df_train.isnull().sum()}\n")

print(f"Null values if train set: \n\n{df_test.isnull().sum()}\n")
mask = np.triu(df.corr())

corr = df.corr()

plt.figure(figsize=(12,6))

sns.heatmap(corr, annot = True,vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', fmt='.1g',mask=mask)

plt.show()
print(f"Count of null values = {df.PassengerId.isnull().sum()}")

print(f"Feature data type: {df.PassengerId.dtypes}")

print(f"Percentage of unique values = {len(df.PassengerId.unique())/len(df)*100}%")
print(f"Count of null values = {df.Survived.isnull().sum()}")

print(f"Feature data type: {df.Survived.dtypes}\n")

fig = make_subplots(rows=1, cols=3,  specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=["Died", "Lived"],  values=df.groupby("Survived")["Survived"].count(), pull=[0.1, 0.1]), 1, 1)

colors = ['DarkBlue', 'ForestGreen']

fig.update_traces(hole=.4, textinfo='label+percent',  hoverinfo="label+percent", marker=dict(colors=colors, line=dict(color='#000000', width=1)))

fig.update_layout(autosize = True, width = 800, height = 300, margin=dict(t=0, b=0, l=0, r=0))

fig.update(layout_showlegend=False)

fig.show()
print(f"Count of null values = {df.Pclass.isnull().sum()}")

print(f"Feature data type: {df.Pclass.dtypes}")

print(f"Correaltion with survival = {round(df.Pclass.corr(df.Survived),2)}\n")

fig = make_subplots(rows=1, cols=3,  specs=[[{'type':'domain'}, {'type':'xy'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=["1 Class", "2 Class", "3 Class"],  values=df.groupby("Pclass")["Pclass"].count(), pull=[0.1, 0.1]), row=1, col=1)

y0 =df.loc[df.Survived==0].groupby("Pclass")["Pclass"].count()

y1 =df.loc[df.Survived==1].groupby("Pclass")["Pclass"].count()

t0 = df.loc[df.Survived==0].groupby("Pclass")["Pclass"].count()/ df.groupby("Pclass")["Pclass"].count()

t1 = df.loc[df.Survived==1].groupby("Pclass")["Pclass"].count()/ df.groupby("Pclass")["Pclass"].count()

fig.add_trace(go.Bar(name='Died', x=["1 Class", "2 Class", "3 Class"], y=y0, text = t0, textposition='auto'), row=1, col=2)

fig.add_trace(go.Bar(name='Lived', x=["1 Class", "2 Class", "3 Class"], y=y1, text = t1, textposition='auto'), row=1, col=2)

colors = ['Thistle', 'Lavender', 'Cornsilk']

fig.update_traces(col = 1, hole=.4, textinfo='label+percent',  hoverinfo="label+percent", marker=dict(colors=colors, line=dict(color='#000000', width=1)))

fig.update_traces(col = 2, texttemplate='%{text:.2%%}', textposition='inside')



fig.update_layout(autosize = True, width = 800, height = 300, margin=dict(t=0, b=0, l=0, r=0))

fig.update(layout_showlegend=False)

fig.update_layout(barmode='stack')



fig.show()
print(f"Count of null values = {df.PassengerId.isnull().sum()}")

print(f"Feature data type: {df.PassengerId.dtypes}")

print(f"Percentage of unique values = {len(df.PassengerId.unique())/len(df)*100}%")
df["Name_prefix"] = df.Name.apply(lambda x: x.split(",")[1].split(".")[0])

df["test_N"] = ce.OrdinalEncoder(return_df=True).fit_transform(df.Name_prefix) #Ordinal Encoder is used only to see some correlation

print(f"Name prefix corr. with survival = {round(df.test_N.corr(df.Survived),3)}")
df.Sex = df.Sex.apply(lambda x: 1 if x == "male" else 0)

print(f"Count of null values = {df.Sex.isnull().sum()}")

print(f"Feature data type: {df.Sex.dtypes}")

print(f"Correaltion with survival = {round(df.Sex.corr(df.Survived),2)}")

fig = make_subplots(rows=1, cols=3,  specs=[[{'type':'domain'}, {'type':'xy'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=["Female", "Male"],  values=df.groupby("Sex")["Sex"].count(), pull=[0.1, 0.1]), row=1, col=1)

y0 =df.loc[df.Survived==0].groupby("Sex")["Sex"].count()

y1 =df.loc[df.Survived==1].groupby("Sex")["Sex"].count()

t0 = df.loc[df.Survived==0].groupby("Sex")["Sex"].count()/ df.groupby("Sex")["Sex"].count()

t1 = df.loc[df.Survived==1].groupby("Sex")["Sex"].count()/ df.groupby("Sex")["Sex"].count()

fig.add_trace(go.Bar(name='Died', x=["Female", "Male"], y=y0, text = t0, textposition='auto'), row=1, col=2)

fig.add_trace(go.Bar(name='Lived', x=["Female", "Male"], y=y1, text = t1, textposition='auto'), row=1, col=2)

colors = ['Thistle', 'Lavender']

fig.update_traces(col = 1, hole=.4, textinfo='label+percent',  hoverinfo="label+percent", marker=dict(colors=colors, line=dict(color='#000000', width=1)))

fig.update_traces(col = 2, texttemplate='%{text:.2%%}', textposition='inside')



fig.update_layout(title="Sex Counts and Survival", autosize = True, width = 800, height = 300, margin=dict(t=100, b=0, l=0, r=0))

fig.update(layout_showlegend=False)

fig.update_layout(barmode='stack')

fig.show()



fig = px.bar(df, x="Sex", y="Survived",  facet_col="Pclass", category_orders={"Pclass": [1, 2, 3]})

fig.update_layout(title="Survival by Sex and Class",autosize = True, width = 525, height = 400, margin=dict(t=100, b=0, l=0, r=0))

fig.show()
print(f"Count of null values = {df.Age.isnull().sum()}")

print(f"Feature data type: {df.Age.dtypes}")

print(f"Correaltion with survival = {round(df.Age.corr(df.Survived),2)}")

fig = px.histogram(df, x="Age", y="PassengerId",  facet_col="Survived",color = "Pclass", facet_row="Sex")

fig.update_layout(title="Distribution of Age by Pclass, Sex and Survived",autosize = True, width = 925, height = 800, margin=dict(t=100, b=0, l=0, r=0))

fig.show()
print(f"Correaltion of Age with survival = {round(df.Age.corr(df.Survived),2)}")

df["Age_groups"] = pd.cut(df.Age, bins = [0,2,16,60,100], labels = [0,1,2,3])

print(f"Correaltion of Age_groups with survival = {round(df.Age_groups.corr(df.Survived),2)}")
imputer = KNNImputer()

imputer.fit(df[["Age", "Pclass", "Sex"]])

df["Age"] = imputer.transform(df[["Age", "Pclass", "Sex"]])[:,0]

df["Age_groups"] = pd.cut(df.Age, bins = [0,2,16,60,100], labels = [0,1,2,3])

print(f"Correaltion of Age_groups with no missing values= {round(df.Age_groups.corr(df.Survived),2)}")
print(f"SibSp \nCount of null values = {df.SibSp.isnull().sum()}")

print(f"Feature data type: {df.SibSp.dtypes}")

print(f"Correaltion with survival = {round(df.SibSp.corr(df.Survived),2)}\n")



print(f"Parch \nCount of null values = {df.Parch.isnull().sum()}")

print(f"Feature data type: {df.Parch.dtypes}")

print(f"Correaltion with survival = {round(df.Parch.corr(df.Survived),2)}")



fig = px.histogram(df, x="SibSp", y="PassengerId",  facet_col="Survived")

fig.update_layout(title="Distribution of SibSp by Survival",autosize = True, width = 925, height = 500, margin=dict(t=100, b=0, l=0, r=0))

fig.show()



fig = px.histogram(df, x="Parch", y="PassengerId",  facet_col="Survived")

fig.update_layout(title="Distribution of Parch by Survival",autosize = True, width = 925, height = 500, margin=dict(t=100, b=0, l=0, r=0))

fig.show()



df["Relatives"] = df.SibSp +df.Parch

fig = px.histogram(df, x="Relatives", y="PassengerId",  facet_col="Survived")

fig.update_layout(title="Distribution of Relatives by Survival",autosize = True, width = 925, height = 500, margin=dict(t=100, b=0, l=0, r=0))

fig.show()
def rel_bins(r):

    if r == 1:

        x = 1

    elif r == 2:

        x = 2

    elif r == 3:

        x = 3

    elif r < 8:

        x = 4

    else:

        x = 5

    return x



df["Relatives_groups"] = df.Relatives.apply(lambda x: rel_bins(x))

print(f"Correaltion with survival = {round(df.Relatives_groups.corr(df.Survived),2)}")
print(f"Count of null values = {df.Ticket.isnull().sum()}")

print(f"Feature data type: {df.Ticket.dtypes}")

df.Ticket.value_counts().nlargest(15)
def ticket_split(s):

    """Split string into numerical and text values"""



    head = s.rstrip('0123456789')

    tail = s[len(head):]

    return head, tail
df["Ticket_letter"] = df.Ticket.apply(lambda x: ticket_split(x)[0])

df["Ticket_number"] = df.Ticket.apply(lambda x: int(ticket_split(x)[1]) if ticket_split(x)[1]!= "" else 0)
df[["Ticket", "Ticket_letter","Ticket_number"]].head(10)
df["test_tkt"] = ce.OrdinalEncoder(return_df=True).fit_transform(df.Ticket_letter) #Ordinal Encoder is used only to see some correlation

print(f"Ticket letter corr. with survival = {round(df.test_tkt.corr(df.Survived),3)}")

print(f"Ticket number corr. with survival = {round(df.Ticket_number.corr(df.Survived),3)}")
print(f"Count of null values = {df.Fare.isnull().sum()}")

print(f"Feature data type: {df.Fare.dtypes}")

print(f"Correaltion with survival = {round(df.Fare.corr(df.Survived),2)}")

df.Fare.hist()

plt.show()

fig = px.box(df,x = "Survived", y="Fare", points="all")

fig.show()



fig = px.histogram(df, x="Fare", color = "Survived", facet_col="Pclass", category_orders={"Pclass": [1, 2, 3]})

fig.update_layout(title="Distribution of Fare by Survival and Pclass",autosize = True, width = 1100, height = 500, margin=dict(t=100, b=0, l=0, r=0))



fig.show()



fig = px.histogram(df, x="Fare", color = "Survived", facet_col="Age_groups", category_orders={"Age_groups": [0, 1, 2, 3]})

fig.update_layout(title="Distribution of Fare by Survival and Age_groups",autosize = True, width = 1100, height = 500, margin=dict(t=100, b=0, l=0, r=0))

fig.update_yaxes(matches=None)

fig.show()

df["Fare_groups"] = pd.cut(df.Fare, bins = [0,50,100,200,300,700], labels = [0,1,2,3,4])

df.Fare_groups.corr(df.Survived)
print(f"Count of null values = {df.Cabin.isnull().sum()}")

print(f"Feature data type: {df.Cabin.dtypes}")

df.Cabin.value_counts().nlargest(15)
def cab_split(s):

    """Split string into numerical and text values"""



    if len(s.split(" "))>1:

        s = s.split(" ")[0]

    head = s.rstrip('0123456789')

    tail = s[len(head):]

    return head, tail
df.Cabin = df.Cabin.fillna("?")

df["Cabin_letter"] = df.Cabin.apply(lambda x: cab_split(x)[0])

df["Cabin_number"] = df.Cabin.apply(lambda x: int(cab_split(x)[1]) if cab_split(x)[1]!= "" else 0)
df[["Cabin", "Cabin_letter","Cabin_number"]].head(10)
df["Cab_count"]= df.Cabin.apply(lambda x: len(x.split(" ")) if x.split(" ")[0]!= "F" else 1)

print(f"Cabin count corr. with survival = {round(df.Cab_count.corr(df.Survived),3)}")
df["test_Cabin"] = ce.OrdinalEncoder(return_df=True).fit_transform(df.Cabin_letter) #Ordinal Encoder is used only to see some correlation

print(f"Cabin letter corr. with survival = {round(df.test_Cabin.corr(df.Survived),3)}")

print(f"Cabin number corr. with survival = {round(df.Cabin_number.corr(df.Survived),3)}")
print(f"Count of null values = {df.Embarked.isnull().sum()}")

print(f"Feature data type: {df.Embarked.dtypes}")

df["test_E"] = ce.OrdinalEncoder(return_df=True).fit_transform(df.Embarked) #Ordinal Encoder is used only to see some correlation

print(f"Embarked corr. with survival = {round(df.test_E.corr(df.Survived),3)}")
fig = px.histogram(df, x="Embarked", color = "Survived", facet_col="Pclass")

fig.show()
df_train.Sex = df_train.Sex.apply(lambda x: 1 if x == "male" else 0) #necessary for inputting

df_test.Sex = df_test.Sex.apply(lambda x: 1 if x == "male" else 0) 

imputer = KNNImputer()

imputer.fit(df_train[["Age", "Pclass", "Sex"]])



for df_t in dfs:



    #Cabin

    df_t.Cabin = df_t.Cabin.fillna("?")



    #Age

    df_t.Age = imputer.transform(df_t[["Age", "Pclass", "Sex"]])[:,0]



    #Embarked

    df_t.Embarked = df_t.Embarked.fillna(df_train.Embarked.value_counts().index[0])



    #Fare

    df_t.Fare = df_t.Fare.fillna(df_train.Fare.median())
for df_t in dfs:

    df_t["Cab_count"]= df_t.Cabin.apply(lambda x: len(x.split(" ")) if x.split(" ")[0]!= "F" else 1)

    df_t["Cabin_letter"] = df_t.Cabin.apply(lambda x: cab_split(x)[0])

    df_t["Cabin_number"] = df_t.Cabin.apply(lambda x: int(cab_split(x)[1]) if cab_split(x)[1]!= "" else 0)

    df_t["Ticket_letter"] = df_t.Ticket.apply(lambda x: ticket_split(x)[0])

    df_t["Ticket_number"] = df_t.Ticket.apply(lambda x: int(ticket_split(x)[1]) if ticket_split(x)[1]!= "" else 0)

    df_t["Name_prefix"] = df_t.Name.apply(lambda x: x.split(",")[1].split(".")[0])

    df_t["Age_groups"] = pd.cut(df_t.Age, bins = [0,2,16,60,100], labels = [0,1,2,3])

    df_t["Relatives"] = df_t.SibSp +df_t.Parch

    df_t["Relatives_groups"] = df_t.Relatives.apply(lambda x: rel_bins(x))

sc = MinMaxScaler().fit(df_train["Fare"].values.reshape(-1, 1))

for df_t in dfs:

    df_t["Fare"] = sc.transform(df_t["Fare"].values.reshape(-1, 1))
enc = ce.OneHotEncoder(return_df=True).fit(df_train.Embarked)

df_emb_train = enc.transform(df_train.Embarked)

df_emb_test = enc.transform(df_test.Embarked)



enc = ce.OneHotEncoder(return_df=True).fit(df_train.Name_prefix)

df_name_train = enc.transform(df_train.Name_prefix)

df_name_test = enc.transform(df_test.Name_prefix)



enc = ce.OneHotEncoder(return_df=True).fit(df_train.Cabin_letter)

df_cab_train = enc.transform(df_train.Cabin_letter)

df_cab_test = enc.transform(df_test.Cabin_letter)



enc = ce.HashingEncoder(n_components=10, return_df=True).fit(df_train.Name_prefix)

df_tkt_train = enc.transform(df_train.Name_prefix)

df_tkt_train.columns = ["ticket_letter_" + str(i) for i in range(10)]

df_tkt_test = enc.transform(df_test.Name_prefix)

df_tkt_test.columns = ["ticket_letter_" + str(i) for i in range(10)]





X = df_train[["Pclass", "Fare", "Age_groups", "Sex", "Relatives_groups"]]

result_train = pd.concat([X, df_emb_train,  df_name_train, df_cab_train, df_tkt_train], axis=1, sort=False)

X = df_test[["Pclass", "Fare", "Age_groups", "Sex", "Relatives_groups"]]

result_test = pd.concat([X, df_emb_test, df_name_test, df_cab_test, df_tkt_test], axis=1, sort=False)

from sklearn.svm import LinearSVC, NuSVC, SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

X = result_train

y = df_train.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf =  SVC(C = 12,  kernel = 'rbf')

clf.fit(X, y)

pred = clf.predict (X_test)

print(f"Accuracy = {round(accuracy_score(y_test, pred),4)}, f1 = {round(f1_score(y_test, pred),4)} \n")



all_accuracies = cross_val_score(estimator=clf, X=X, y=y, cv=5)

print(f"Accuracy mean from 5 fold cross validation = {round(all_accuracies.mean(),4)}\n")

print("Confusion matrix:\n\n",confusion_matrix(y_test, pred))
pred = clf.predict(result_test)

sumission_df = pd.DataFrame(data={'PassengerId': df_test.PassengerId.values, 'Survived': pred})

print(sumission_df.head())

sumission_df.to_csv ('submission.csv', index = False, header=True, sep = ",")