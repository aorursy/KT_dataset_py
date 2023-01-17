import time

start_time = time.time()



import pandas as pd 

import numpy as np



from scipy.sparse import csr_matrix, hstack



from sklearn.linear_model import Ridge

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer



def replaceMissing(dataset):

    dataset['category_name'].fillna(value='MV99', inplace=True)

    dataset['brand_name'].fillna(value='MV99', inplace=True)

    dataset['item_description'].fillna(value='MV99', inplace=True)

    

def createCategories(dataset):

    cat_lists = dataset['category_name'].apply(lambda x: x.split('/'))

    cat_df = pd.DataFrame(cat_lists.tolist(), columns=['category1','category2','category3','category4','category5'])

    ds = pd.concat([dataset, cat_df], axis=1)

    return ds



def cleanCategories(ds1, ds2, minCount): 

    for var in ['category1','category2','category3','category4','category5']:

        

        ds1[var].fillna(value='MV99', inplace=True)

        ds2[var].fillna(value='MV99', inplace=True)

        

        keep = ds1[var].value_counts()[

            (ds1[var].value_counts() > minCount) & (ds2[var].value_counts() > minCount)

        ].index.tolist()

        

        ds1[var] = ds1[var].apply(lambda x: x if x in keep else 'SV01')

        ds2[var] = ds2[var].apply(lambda x: x if x in keep else 'SV01')

        

def cleanBrands(ds1, ds2, minCount):   

    keepBrands = ds1['brand_name'].value_counts()[

        (ds1['brand_name'].value_counts() > minCount) & (ds2['brand_name'].value_counts() > minCount)

    ].index.tolist() 

    

    ds1['brand'] = ds1['brand_name'].apply(lambda x: x if x in keepBrands else 'SV01')

    ds2['brand'] = ds2['brand_name'].apply(lambda x: x if x in keepBrands else 'SV01')

    

train = pd.read_csv('../input/train.tsv',sep='\t')

test = pd.read_csv("../input/test.tsv", sep='\t')



print("{} second to finish import".format(time.time() - start_time))



full = [train,test]



submission = test['test_id']



full2 = []



for df in full:

    replaceMissing(df)

    df2 = createCategories(df)

    full2.append(df2)

    

train2 = full2[0]

test2 = full2[1]



cleanCategories(train2, test2, 10)

cleanBrands(train2, test2, 10)



print("{} seconds to finish cleaning data".format(time.time() - start_time))



nrow_train = train.shape[0]



traintest = pd.concat([train2.drop(['train_id','price'], axis=1), test2.drop(['test_id'], axis=1)])



lb = LabelBinarizer(sparse_output=True)



X_c1 = lb.fit_transform(traintest['category1'])

X_c2 = lb.fit_transform(traintest['category2'])

X_c3 = lb.fit_transform(traintest['category3'])

X_c4 = lb.fit_transform(traintest['category4'])

X_c5 = lb.fit_transform(traintest['category5'])

X_brand = lb.fit_transform(traintest['brand'])



print("{} seconds to finish LabelBinerizer".format(time.time()-start_time))



tv = TfidfVectorizer(max_features=50000, ngram_range=(1,3), stop_words='english')

X_description = tv.fit_transform(traintest['item_description'])



X_dummies = csr_matrix(pd.get_dummies(traintest[['item_condition_id', 'shipping']],sparse=True).values)



sparse_merge = hstack((X_dummies,X_description,X_brand,X_c1,X_c2,X_c3,X_c4,X_c5)).tocsr()



print("{} seconds to finish Sparse Merge".format(time.time()-start_time))



X = sparse_merge[:nrow_train]

X_test = sparse_merge[nrow_train:]



y = np.log1p(train['price'])

end_time = time.time()

print("Data Processing took {} seconds".format(end_time - start_time))
model = Ridge(alpha=1, solver='sag', fit_intercept=True, random_state=111)

model.fit(X,y)

preds = np.expm1(model.predict(X_test))

submission = pd.concat([test['test_id'],pd.DataFrame(preds, columns=['price'])],axis=1)

submission.to_csv("submission_simple_ridge_1.csv", index=False)