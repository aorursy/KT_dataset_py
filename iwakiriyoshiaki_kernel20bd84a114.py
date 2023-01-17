import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



# testとtrainをまとめる

y_train = train.iloc[:,1] #正解ラベルの用意

train = train.drop('Survived',axis=1)

df = pd.concat([train,test])
#データの前処理

df.shape



df.isnull().sum()
df = df.drop('Ticket',axis=1).drop('Cabin',axis=1)

df["Embarked"].value_counts()#値の中身を知りたいときvalue_counts()
#欠損値の補完

df["Embarked"].fillna("S",inplace=True)#値の多い”S”で補完

df["Fare"].fillna(df.Fare.mean(), inplace=True)#Fareは平均値で補完（DataFrame.Column.mean()←カラム名は[]等で囲わない）
#新たな特徴量の生成←行列の次元数を下げるためあたらしい特徴量にまとめ、既存のカラムを削除する。（機械学習には次元数が少ない方が良い）

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df=df.drop('SibSp',axis=1).drop('Parch',axis=1)
"""

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

%matplotlib inline



df2 = pd.concat([train,test],sort=True)



df2 = df2.drop('Ticket',axis=1).drop('Cabin',axis=1)

df2["Embarked"].fillna("S",inplace=True)

df2["Fare"].fillna(df2.Fare.mean(), inplace=True)

df2['FamilySize'] = df2['SibSp'] + df2['Parch'] + 1



emb_dum = pd.get_dummies(df2['Embarked'])

df2 = pd.concat((df2,emb_dum),axis=1)



df2["Embarked"] = df2['S']

df2 = df2.drop(['C', 'Q', 'S'],axis=1)



sex_dum = pd.get_dummies(df2['Sex'])

df2 = pd.concat((df2, sex_dum),axis=1)



df2["sex"] = df2.male

df2 = df2.drop('female',axis=1)

df2 = df2.drop('male',axis=1)



#ヒートマップの作りかた（相関係数）!!

corr_mat = df2.corr()#相関係数を導出

sns.heatmap(corr_mat,

            vmin=-1.0,

            vmax=1.0,

            center=0,

            fmt='.1f',

            xticklabels=corr_mat.columns.values,

            yticklabels=corr_mat.columns.values

           )

plt.show()

"""
df
# 性別(sex)と乗船場所(enmark)のデータはダミー変数を作成する

# まずはsexから、性別のdummy変数を作成する

sex_dum = pd.get_dummies(df['Sex'])

# データの連結をする

df = pd.concat((df,sex_dum),axis=1)

# データからsexの列を取り除く

df = df.drop('Sex',axis=1)

# 性別は男か女かのみなので男なら1というデータのみあればよい

df = df.drop('female',axis=1)



# 乗船場所でも同様

emb_dum = pd.get_dummies(df['Embarked'])

df = pd.concat((df,emb_dum),axis=1)

df = df.drop('Embarked',axis=1)

df = df.drop('S',axis=1)
def name_classifier(name_df):    #名前ごとの分類（より正確な平均年齢を求めるため）

    name_class_df = pd.DataFrame(columns={'miss','mrs','master','mr'})



    for name in name_df:        

        if 'Miss' in name:

            df = pd.DataFrame([[1,0,0,0]],columns={'miss','mrs','master','mr'})

        elif 'Mrs' in name:

            df = pd.DataFrame([[0,1,0,0]],columns={'miss','mrs','master','mr'})

        elif 'Master' in name:

            df = pd.DataFrame([[0,0,1,0]],columns={'miss','mrs','master','mr'})

        elif 'Mr' in name:

            df = pd.DataFrame([[0,0,0,1]],columns={'miss','mrs','master','mr'})

        else :

            df = pd.DataFrame([[0,0,0,0]],columns={'miss','mrs','master','mr'})

        name_class_df = name_class_df.append(df,ignore_index=True)        

    return name_class_df
name = df.iloc[:,2]

name_class = name_classifier(name)



name_class.head()
def ave_age(df,df_name):#平均年齢を出す関数

    miss=0

    mrs=0

    master=0

    mr=0

    c_miss=0

    c_mrs=0

    c_master=0

    c_mr=0

    for i in range(1309):#年齢がNULLでないものについて、合計数を求める。

        if df.isnull().iloc[i,3]==False:

            if df_name.iloc[i,0]==1:

                miss=miss+df.iloc[i,3]

                c_miss=c_miss+1

            if df_name.iloc[i,1]==1:

                mrs=mrs+df.iloc[i,3]

                c_mrs=c_mrs+1

            if df_name.iloc[i,2]==1:

                master=master+df.iloc[i,3]

                c_master=c_master+1

            if df_name.iloc[i,3]==1:

                mr=mr+df.iloc[i,3]

                c_mr=c_mr+1

    return [miss/c_miss,mrs/c_mrs,master/c_master,mr/c_mr]#平均値をリストで返す
#求めた平均値で欠損値を埋める

for i in range(1309):

    if df.isnull().iloc[i,3]==True:

        if name_class.iloc[i,0]==1:

            df.iloc[i,3]=21.8

        elif name_class.iloc[i,1]==1:

            df.iloc[i,3]=37

        elif name_class.iloc[i,2]==1:

            df.iloc[i,3]=5.5

        if name_class.iloc[i,3]==1:

            df.iloc[i,3]=32.3

            

df.isnull().sum()
df['Age'].fillna(df['Age'].median(),inplace=True)

df=df.drop('Name',axis=1)#Nameはもう使わないので、削除
df["Fare"].fillna(df.Fare.median(), inplace=True)
train_data=df.iloc[0:891,:]

test_data=df.iloc[891:,:]



id_test=test_data.iloc[:,0] #提出用データに必要です



x_train=train_data.drop('PassengerId',axis=1)

x_test=test_data.drop('PassengerId',axis=1)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier()



from sklearn.model_selection import GridSearchCV

param = {'n_estimators':[10,100,500,1000],'max_depth':[3,6,12],'criterion':['gini','entropy'],'random_state':[7]}

grid_forest = GridSearchCV(forest,param) #defaultではcv=3なのでデータを3つに分け、3回交差検証が行われます

grid_forest.fit(x_train,y_train)
grid_forest.best_params_# グリッドサーチにて、最適なハイパーパラメータの値を導出
best_forest=RandomForestClassifier(max_depth= 6, n_estimators=10, criterion='entropy',random_state=7)

best_forest.fit(x_train,y_train)
result = np.array(best_forest.predict(x_test))

df_result=pd.DataFrame(result,columns=['Survived'])

df_result=pd.concat([id_test,df_result],axis=1)

df_result['Survived'] = np.array(round(df_result['Survived']), dtype='int')

df_result.to_csv('reult.csv', index=False)