#LOADING DOC2VEC AND CSV FILES



import pandas as pd

import numpy as np

from gensim.models import Doc2Vec



sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")



model = Doc2Vec.load('/kaggle/input/doc2vec-english-binary-file/doc2vec.bin')

def doc2vec(text):

    return model.infer_vector(text.split())
#BUILDING TRAIN DATASET



data = []

for index, row in train.iterrows():

    keyword_vec = doc2vec(str(row['keyword']).lower())

    location_vec = doc2vec(str(row['location']).lower())

    text_vec = doc2vec(row['text'].lower())

    data.append([row['id'], np.concatenate((keyword_vec, location_vec, text_vec), axis=0, out=None), row['target']])



df = pd.DataFrame(data, columns = ['id', 'vector','target'])

df.to_csv('/kaggle/working/vectorized_dataset.csv',index=False)
#TRAINING MODEL



from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Normalizer

from sklearn import svm



scaler = StandardScaler()



X = list(df['vector'])

y = list(df['target'])



#scaler.fit(X)

#X = scaler.transform(X)



normalizer = Normalizer().fit(X)

X = normalizer.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)



degrees = [2,3,4,5,6,7,8,9]

Cs = [1,10,100]

kernels = ['poly','rbf']



for degree in degrees:

    for C in Cs:

        print("Degree: "+str(degree)+" C: "+str(C))

        #clf = LogisticRegression(solver="lbfgs").fit(X_train, y_train)

        #clf = MLPClassifier(activation='relu', alpha=0.00001, hidden_layer_sizes=(300), max_iter = 300, solver="lbfgs").fit(X_train,y_train)

        clf = svm.SVC(kernel='poly', degree=degree, gamma=1, max_iter=10000, C=C, tol=1e-3).fit(X_train, y_train)

        print("Train_Score: "+str(clf.score(X_train, y_train)))

        print("Test_Score: "+str(clf.score(X_test, y_test)))



#Best setting: degree=6 kernel=poly and C=1

clf = svm.SVC(kernel='poly', degree=degree, gamma=1, max_iter=10000, C=C, tol=1e-3).fit(X, y)

#BUILDING TEST DATASET



data_test = []

for index, row in test.iterrows():

    keyword_vec = doc2vec(str(row['keyword']).lower())

    location_vec = doc2vec(str(row['location']).lower())

    text_vec = doc2vec(row['text'].lower())

    data_test.append([row['id'], np.concatenate((keyword_vec, location_vec, text_vec), axis=0, out=None)])



df_test = pd.DataFrame(data_test, columns = ['id', 'vector'])

X_test = list(df_test['vector'])

X_test = normalizer.transform(list(df_test['vector']))
#PREDICTION AND RESULTS



result = list(clf.predict(X_test))



to_send = []

for i,id in enumerate(df_test["id"]):

    to_send.append([id,result[i]])



df_result = pd.DataFrame(to_send, columns=['id','target'])

df_result.to_csv('/kaggle/working/simple_results_SVM_Normalization.csv',index=False)