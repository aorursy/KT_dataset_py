!pip install pydotplus
import pandas as pd

import numpy as np

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



print('-'*136)

train.head(5)
# Used for check columes

# train.columns
train.info()
train.isnull().any()
print('Train columns with null values:\n', train.isnull().sum())

print("-"*30)
# Basic Info about Analisys (Only show the number coefficient rows)

train.describe()
test.head()
test.describe()
# train['Parch'].unique()
#for dev_data in train:    

    #complete missing age with median

#    dev_data['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

#    dev_data['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

#    dev_data['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

#delete the cabin feature/column and others previously stated to exclude in train dataset

#drop_column = ['Name', 'Sex', 'Ticket','Cabin','PassengerId']

#dev_data.drop(drop_column, axis=1, inplace = True)
dev_data = train.drop(['Name', 'Ticket','Cabin','PassengerId'], axis=1)

dev_data.head(10)
print(train['Age'].mean())

dev_data = dev_data.fillna({'Age':train['Age'].mean() })

dev_data = dev_data.fillna({'Embarked': 'S'})

dev_data.isnull().any()

dev_data.head(10)
# Convert data to one-hot

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



df_copy = dev_data.copy()  # 結果格納用にdfをコピー

le_list = []         # LabelEncoderのオブジェクト格納用

dummy_name = []      # ダミー変数の列名用



# LabelEncoderによる定性データの数値化

# 注意：LebelEncoderは1列ごとにfit_transformを行う必要があります

for i in ['Sex','Embarked']:

    le = LabelEncoder()

    df_copy[i] = le.fit_transform(dev_data[i])

    

    le_list = np.append(le_list, le)

    dummy_name = np.append(dummy_name, i+'_'+le.classes_)

    

df_copy.head(10)
df_copy['label'] = df_copy['Survived']

dev_data = df_copy.drop(['Survived'], axis=1)

dev_data.head()
# Standarlization



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

dev_data.iloc[:, 0:-1] = sc.fit_transform(dev_data.iloc[:, 0:-1])

dev_data.head()
from sklearn.model_selection import train_test_split

#特徴変数と目的変数の抽出

X_cl = dev_data.iloc[:,0:-1]

#print(X_cl)

y_cl = dev_data['label'].astype(str)

#y_cl = y_cl.tostring()

X_cl_train, X_cl_test, y_cl_train, y_cl_test = train_test_split(X_cl, y_cl, test_size=0.2)
X_cl_train.describe()
X_cl_test.describe()
# ライブラリのインポート

import sklearn.tree as tree

import pydotplus

from IPython.display import Image

from sklearn.externals.six import StringIO
# 決定木

clr_tree = tree.DecisionTreeClassifier(max_depth=2) # 決定木の呼び出し

clr_tree.fit(X_cl_train, y_cl_train)                # 決定木の適用（学習）

clr_tree.feature_importances_                       # 重要度の取得

pd.DataFrame(clr_tree.feature_importances_, index=X_cl.columns, columns=['importances'])  # 重要度の取得（見栄えがいいVer）
#決定木の可視化



features = X_cl.columns         # 特徴量の列名を変数に格納

targets = np.unique(y_cl)       # 目的変数の値を変数に格納.(変数の中身を重複排除)



dot_tmp=StringIO()

tree.export_graphviz(clr_tree,  # 作成したモデルを指定。可視化に使うルールを、テキスト（dot形式）として変数に格納

                     out_file=dot_tmp,

                     feature_names=features,

                     class_names=targets,

                     filled=True,

                     rounded=True,

                     special_characters=True)



graph = pydotplus.graph_from_dot_data(dot_tmp.getvalue()) # テキスト(dot形式)からグラフ化し、pngイメージ化

Image(graph.create_png())

graph.write_png('tree.png')
# Evaluate Desision Tree

# ライブラリのインポート

from sklearn.metrics import precision_score, recall_score, make_scorer, confusion_matrix, f1_score



#各学習手法(model)にテストデータ(X,y)を入力した場合の混合行列、正解率、適合率、再現率を表示

def evaluate_cl(model, X, y):

    y_pred = model.predict(X)

    display(confusion_matrix(y_true=y, y_pred=y_pred))

    display('正解率：'+str(model.score(X,y)))

    display('適合率：'+str(precision_score(y_true=y, y_pred=y_pred, pos_label='1')))

    display('再現率：'+str(recall_score(y_true=y, y_pred=y_pred, pos_label='1')))

    display('F値：'+str(f1_score(y_true=y, y_pred=y_pred, pos_label='1')))

    

evaluate_cl(clr_tree,X_cl_test,y_cl_test)
# RandomForestClassifier

# ライブラリのインポート

from sklearn.ensemble import RandomForestClassifier



#ランダムフォレスト

rf = RandomForestClassifier(n_estimators=1000, max_features=4, random_state=1, n_jobs=-1) # 手法の呼び出し 決定木の数は1000個 

rf.fit(X_cl_train, y_cl_train)  # ランダムフォレストの適用

rf.feature_importances_         # 重要度の取得

pd.DataFrame(clr_tree.feature_importances_, index=X_cl.columns, columns=['importances'])  # 重要度の取得（見栄えがいいVer）



#ランダムフォレストの評価

evaluate_cl(rf,X_cl_test,y_cl_test)
# LogisticRegression with nopenality

from sklearn.linear_model import LogisticRegression

lr_l0 = LogisticRegression(random_state=0).fit(X_cl_train, y_cl_train)

lr_l0.coef_

evaluate_cl(lr_l0,X_cl_test,y_cl_test)
lr_l1 = LogisticRegression(C=0.01, penalty='l1', tol=0.01, solver='saga')

lr_l1.fit(X_cl_train, y_cl_train)

lr_l1.coef_

evaluate_cl(lr_l1,X_cl_test,y_cl_test)
#L2正規化 正則化強め

lr_l2 = LogisticRegression(penalty='l2', C=0.01)

lr_l2.fit(X_cl_train, y_cl_train)

lr_l2.coef_



#ロジスティック回帰（L1正則化）の評価

evaluate_cl(lr_l2,X_cl_test,y_cl_test)
from sklearn.svm import SVC



clf = SVC(gamma='auto')

clf.fit(X_cl_train, y_cl_train)

#print(clf.predict(X_cl_test))

#print(gender_submission)       <=============== This data is used for ????
evaluate_cl(clf,X_cl_test,y_cl_test)
# MLP Classifier

from sklearn.neural_network import MLPClassifier

MPLclf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 3), random_state=1)

MPLclf.fit(X_cl_train, y_cl_train)

evaluate_cl(MPLclf,X_cl_test,y_cl_test)
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_cl_train, y_cl_train)
evaluate_cl(gnb,X_cl_test,y_cl_test)
dev2_data = test.drop(['Name', 'Ticket','Cabin','PassengerId'], axis=1)

dev2_data = dev2_data.fillna({'Age':train['Age'].mean() })

dev2_data = dev2_data.fillna({'Fare':train['Fare'].mean() })

dev2_data = dev2_data.fillna({'Embarked': 'S'})

dev2_data.isnull().any()
# Convert data to one-hot

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



df2_copy = dev2_data.copy()  # 結果格納用にdfをコピー

le_list2 = []         # LabelEncoderのオブジェクト格納用

dummy_name2 = []      # ダミー変数の列名用



# LabelEncoderによる定性データの数値化

# 注意：LebelEncoderは1列ごとにfit_transformを行う必要があります

for i in ['Sex','Embarked']:

    le2 = LabelEncoder()

    df2_copy[i] = le2.fit_transform(dev2_data[i])

    

    le_list2 = np.append(le_list2, le2)

    dummy_name2 = np.append(dummy_name2, i+'_'+le.classes_)

    

df2_copy.head(10)
from sklearn.preprocessing import StandardScaler

sc2 = StandardScaler()

df2_copy.iloc[:, 0:7] = sc2.fit_transform(df2_copy.iloc[:, 0:7])

df2_copy.head()
y_pred = gnb.fit(X_cl_train, y_cl_train).predict(df2_copy)

#print(y_pred)

#df_pred = pd.DataFrame()

#df_pred.columns = ['a', 'b']

#df_pred['a'] = y_pred



d = {'ID': test['PassengerId'] , 'predict':y_pred }

df_pred= pd.DataFrame(data=d)

print(df_pred.head(30))