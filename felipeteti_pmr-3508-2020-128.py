import pandas as pd 

import matplotlib.pyplot as plt 

import numpy as np

import seaborn as sns 

plt.style.use('ggplot')
df = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values='?')

df.head()
df.drop("Id", inplace=True, axis=1)
df.info()
df.describe()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



df.income = le.fit_transform(df.income)
sns.pairplot(df, hue="income", plot_kws={"alpha":0.5},

                diag_kws={'bw':"1.0"})

plt.show()
sns.violinplot(x="income", y="hours.per.week", data=df)

plt.show()

print(df.groupby("income")["hours.per.week"].describe())
df_g =  pd.DataFrame(df.groupby("sex").income.value_counts(normalize=True))

df_g.rename(columns={'income': 'count'}, inplace=True)

df_g.reset_index(inplace=True)

sns.barplot(x="sex", y="count", hue="income", data=df_g)

plt.show()
df.groupby("sex").income.value_counts(normalize=True)
df_g =  pd.DataFrame(df.groupby("income").sex.value_counts(normalize=True))

df_g.rename(columns={'sex': 'count'}, inplace=True)

df_g.reset_index(inplace=True)

sns.barplot(x="income", y="count", hue="sex", data=df_g)

plt.show()
df.groupby("income").sex.value_counts(normalize=True)
sns.countplot(x="workclass", data=df)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)

plt.show()
df_g =  pd.DataFrame(df.groupby("workclass").income.value_counts(normalize=True))

df_g.rename(columns={'income': 'count'}, inplace=True)

df_g.reset_index(inplace=True)

df_g.sort_values('count',inplace=True)

sns.barplot(y="workclass", x="count", hue="income", data=df_g)



plt.show()
sns.catplot(y="workclass", x="income", kind="bar", data=df);
sns.distplot(df['age'])

plt.show()

print(df['age'].describe())
sns.violinplot(x="income", y="age", data=df)

plt.show()
sns.countplot(x="race", data=df)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)

plt.show()
sns.catplot(x="race", y="income", kind="bar", data=df);

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)

plt.show()
df_g =  pd.DataFrame(df.groupby("race").income.value_counts(normalize=True))

df_g.rename(columns={'income': 'count'}, inplace=True)

df_g.reset_index(inplace=True)

df_g.sort_values('count',inplace=True)

sns.barplot(y="race", x="count", hue="income", data=df_g)



plt.show()
df.isna().sum()/df.shape[0]*100 #Porcentagem de NaN
sns.countplot(x="workclass", data=df)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)

plt.show()
sns.countplot(x="occupation", data=df)

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)

plt.show()
sns.countplot(x="native.country", data=df)

plt.xticks(

    rotation=90, 

    horizontalalignment='right',

    fontweight='light'  

)

plt.show()
df.drop(['native.country'], axis=1, inplace=True)
from sklearn.impute import SimpleImputer

for i in ["workclass", "occupation"]:

    imp_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent')



    imp_mf.fit(np.array(df[i]).reshape(-1, 1))

    df[i] = imp_mf.transform(np.array(df[i]).reshape(-1, 1))
df.isna().sum()
categorical = ["workclass", "marital.status",  "sex", "race", 

               "occupation", "relationship"]

df = pd.get_dummies(df, columns=categorical, prefix=categorical)

df.head()
from sklearn.preprocessing import StandardScaler



sl = StandardScaler()

df[["hours.per.week", "age", "education.num"]] = sl.fit_transform(df[["hours.per.week",

                                                                                    "age", "education.num"]])

y = df["income"]

X = df.drop(["fnlwgt", "education", "income"], axis =1)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



k_max = None

max_acc=0

for k in range(20, 31):

    

    KNNclf = KNeighborsClassifier(n_neighbors=k, algorithm='brute')

    

    score = cross_val_score(KNNclf, X, y, cv = 5).mean()

    

    print(k," vizinhos; Acurácia:" , round(score,6), ";")

    if score > max_acc:

        k_max = k

        max_acc = score

        

print('\nMelhor k: {}'.format(k_max))
knn = KNeighborsClassifier(n_neighbors=k_max, algorithm='brute')

knn.fit(X, y)
df_teste = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values='?')

categorical = ["workclass", "marital.status",  "sex", "race", 

               "occupation", "relationship"]

df_teste = pd.get_dummies(df_teste, columns=categorical, prefix=categorical) 
from sklearn.preprocessing import StandardScaler



sl = StandardScaler()

df_teste[["hours.per.week", "age", "education.num"]] = sl.fit_transform(df_teste[["hours.per.week",

                                                                                    "age", "education.num"]])



df_teste.drop(["fnlwgt", "education", "native.country", "Id"], inplace=True, axis=1)
X_test= df_teste.values
y_pred = knn.predict(X_test)
pred = pd.DataFrame(columns = ["Id","income"])



pred.Id = df_teste.index

pred.income=y_pred

pred.income.replace({0:"<=50K", 1:">50K" }, inplace=True)

pred.to_csv("submission.csv", index=False)