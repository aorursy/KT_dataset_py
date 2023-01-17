# warningsを無視する
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# 文字のサイズ
plt.rcParams["font.size"] = 18
# サイズの設定
plt.rcParams['figure.figsize'] = (8.0, 6.0)
sns.boxplot(x='Pclass', y='Age', data=df_train)
plt.xticks([0.0,1.0,2.0], ['1st','2nd','3rd'])
plt.title('チケットクラスごとの年齢の箱ひげ図')
plt.xlabel('Pclass(チケットクラス)')
plt.ylabel('Age(年齢)')
# PclassごとにAgeの平均を算出
df_train.groupby('Pclass').mean()['Age'] 
# Ageがnullの場合に、Pclassに応じてAgeに代入する関数
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):        
        if Pclass == 1:
            return 39
        elif Pclass == 2:
            return 30
        else:
            return 25    
    else:
        return Age

# Embarkedの補完
df_train.loc[df_train['PassengerId'].isin([62, 830]), 'Embarked'] = 'C'

# Fareの補完
df_test.loc[df_test['PassengerId'] == 1044, 'Fare'] = 13.675550

data = [df_train, df_test]
for df in data:
    # Ageの補完
    df['Age'] = df[['Age','Pclass']].apply(impute_age, axis = 1) 

    # 性別の変換
    df['Sex'] = df['Sex'].map({"male": 0, "female": 1})
        
    # Embarked
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
df_train.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
df_test.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
X_train = df_train.drop(["PassengerId", "Survived"], axis=1) # 不要な列を削除
Y_train = df_train['Survived'] # Y_trainは、df_trainのSurvived列
X_test  = df_test.drop('PassengerId', axis=1).copy()
from sklearn.ensemble import RandomForestClassifier
# 学習と予測を行う
forest = RandomForestClassifier(n_estimators=10, random_state=1)
forest.fit(X_train, Y_train)
Y_prediction = forest.predict(X_test)
submission = pd.DataFrame({
        'PassengerId': df_test['PassengerId'],
        'Survived': Y_prediction
    })
submission.to_csv('submission.csv', index=False)

forest.feature_importances_
for i,k in zip(X_train.columns,forest.feature_importances_):
    print(i,round(k,4))