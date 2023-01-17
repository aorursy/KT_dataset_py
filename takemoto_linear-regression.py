# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import types

from sklearn.linear_model import LinearRegression

from scipy.sparse import coo_matrix, hstack

from scipy import io



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# pandasのデータフレームを返す

# train_or_testには'train'か'test'を入れる

def load_data(path,train_or_test,brand_threshold = 100,category_threshold = 50,frequent_brands=None,frequent_categories=None):

    data_pd = pd.read_csv(path, error_bad_lines=False, encoding='utf-8', header=0, delimiter='\t')

    #ブランド名がないものを'NO_BRAND'とする

    data_pd['brand_name'] = data_pd['brand_name'].fillna('NO_BRAND')

    data_pd=data_pd.fillna("")



    if train_or_test == 'train':

        frequent_brands = data_pd['brand_name'].value_counts()[data_pd['brand_name'].value_counts()>brand_threshold].index

        frequent_categories = data_pd['category_name'].value_counts()[data_pd['category_name'].value_counts()>category_threshold].index

    elif train_or_test != 'test':

        print('Error : Please input "train" or "test" in train_or_test')

        return

    

    if type(frequent_brands)==type(None) or type(frequent_categories)==type(None):

        print('Error : Please load train data first')

        return

    else:

        data_pd.loc[~data_pd['brand_name'].isin(frequent_brands),'brand_name']= 'SOME_BRAND'

        data_pd.loc[~data_pd['category_name'].isin(frequent_categories),'category_name'] = 'SOME_CATEGORY'

        

    return data_pd,frequent_brands,frequent_categories
csv_train_path = u'../input/train.tsv'

csv_test_path = u'../input/test.tsv'

train_data_pd, frequent_brands, frequent_categories = load_data(csv_train_path,'train',brand_threshold=100,category_threshold=50)

test_data_pd, _, _ = load_data(csv_test_path,'test',frequent_brands=frequent_brands,frequent_categories=frequent_categories)
train_data_pd.iloc[:10]
test_data_pd.iloc[:10]
use_cols = ['item_condition_id','brand_name','shipping','category_name']

train_num = len(train_data_pd)

test_num = len(test_data_pd)
y = np.array(train_data_pd['price'])

y = np.log(y+1)

print(y)
# scipyのsparse matrix(coo_matrix)X_transform と 変数のリストvariables を返す

# save_pathに何も指定しない場合ファイルを保存しない 指定した場合指定したディレクトリ内に保存する

def make_onehot(use_cols,data_pd,train_or_test,save_path=None):

    variables = []

    flag = 0

    for use_col in use_cols:

        dummy_pd = pd.get_dummies(data_pd[use_col]).astype(np.uint8)

        if flag==0:

            X_transform = coo_matrix(dummy_pd.values)

            flag=1

        else:

            X_transform = hstack([X_transform,coo_matrix(dummy_pd.values)])

        

        variables.extend( list( dummy_pd.columns ) )

        

        if save_path is not None:

            if train_or_test != 'test' and train_or_test != 'train':

                print('Error : Please input "train" or "test" in train_or_test')

                return

            save_path_ = '{}/{}_{}.csv'.format(save_path,use_col,train_or_test)

            dummy_pd.to_csv(save_path_,index=False,encoding="utf8")

            

    if save_path is not None:

        # sparse matrixの保存

        io.savemat("{}/X_transform_{}".format(save_path,train_or_test), {"X_transform":X_transform})

        print('sparse matrixを保存しました。次回からはsparse matrixを読み込んで学習に利用してください')



    return X_transform,np.array(variables)
X_transform_train,variables = make_onehot(use_cols,train_data_pd,'train',save_path=None)

print(X_transform_train.toarray()[:5])
X_transform_test,variables_ = make_onehot(use_cols,test_data_pd,'test',save_path=None)

print(X_transform_test.toarray()[:5])
# apply your linear regression as you want

model = LinearRegression()

model.fit(X_transform_train, y)



print("train RMSLE : %.2f" % np.sqrt( np.mean((model.predict(X_transform_train) - y) ** 2) ) )
# 回帰係数

print("Model coef(回帰係数) : {}".format(model.coef_))

 

# 切片 (誤差)

print("Model inteercept(切片 (誤差)) : {}".format(model.intercept_) )

 

# 決定係数

print("Model score(決定係数) : {}".format(model.score(X_transform_train, y)) )

pd.set_option("display.max_rows", 1300)

pd.set_option("display.max_colwidth", 63)

coef = np.array(model.coef_)

print("要素ごとの回帰係数を表示")

dataframe = pd.DataFrame([])

dataframe['variables'] =  variables[np.argsort(coef)[::-1]] 

dataframe['coef'] = np.sort(coef)[::-1]

print(dataframe.iloc[:10])
#testデータを予測

prediction = np.exp(model.predict(X_transform_test))-1
submission = pd.DataFrame([])

submission['test_id'] = test_data_pd['test_id']

submission['price'] = pd.DataFrame(prediction)

submission.to_csv('submission.csv',index=None)

submission.iloc[:10]