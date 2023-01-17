import numpy as np

from sklearn.preprocessing import scale
def other_scores(Y_test,prediction):

    n=prediction.shape[0]

    precision_score=0

    recall_score=0

    F1_score=0

    for i in range(n):

        actual_l=set(Y_test.rows[i])

        predict_l=set(prediction.rows[i])

        if len(predict_l)!=0:

            precision_score+=len(actual_l.intersection(predict_l))/len(predict_l)

        recall_score+=len(actual_l.intersection(predict_l))/len(actual_l)

        F1_score+=2*len(actual_l.intersection(predict_l))/(len(actual_l)+len(predict_l))

    

    return precision_score/n,recall_score/n,F1_score/n

    

        

def label_score(Y_test,pred):

    label_count=[0]*4

    true_pred_count=[0]*4

    for j  in range(4):

        for i in range(len(pred)):

            if Y_test[i][j]==1:

                label_count[j]+=1

            if Y_test[i][j]==1 and pred[i][j]==1:

                true_pred_count[j]+=1

    pred_sum=0

    label_sum=0

    macro_accuracy=[]

    for i in range(4):

        if label_count[i]!=0:

            macro_accuracy.append(true_pred_count[i]/label_count[i])

        else:

            macro_accuracy.append('None')

        pred_sum=pred_sum+true_pred_count[i]

        label_sum=label_sum+label_count[i]

    micro_accuracy=pred_sum/label_sum



X=np.load('../input/benchmark_pssm_feat_multi.npy')

len(X),len(X[1])

labels=np.load("../input/benchmark_pssm_label_multi.npy")

len(labels), len(labels[1])



X=list(X)

labels=list(labels)

Nlabels=len(labels[0])

Ndata=len(X)

X=list(scale(X))

from sklearn.cluster import KMeans

import random



rand_y=[]

rand_x=[]

X_dash=X.copy()

labels_dash=labels.copy()



while len(X_dash)!=0:                            #Shuffling of data

    Random=random.randint(0,len(X_dash)-1)

    rand_x.append(list(X_dash.pop(Random)))

    rand_y.append(labels_dash.pop(Random))

X_plus=[[],[],[],[],[]]

X_neg=[[],[],[],[],[]]



for i in range(Ndata):                         #Partitioning of data based on postives (X_plus) and negatives (X_neg) for each label

    for j in range(Nlabels):

        if rand_y[i][j]==1:

            X_plus[j].append(rand_x[i])

        else :

            X_neg[j].append(rand_x[i])



for i in range(5):

    X_plus[i]=np.array(X_plus[i])

    X_neg[i]=np.array(X_neg[i])

    print("label ",i+1,"postives", X_plus[i].shape, "negatives", X_neg[i].shape)

from sklearn.svm import SVC

#Dataset transformation using cluster analysis

from sklearn.metrics.pairwise import paired_distances

from sklearn.model_selection import LeaveOneOut

#SVM

import numpy as np

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score,jaccard_score

from sklearn.svm import SVC

from scipy.sparse import lil_matrix

for clusters in range(16,17):

    label_5=0

    label_1=0

    label_2=0

    label_3=0

    label_4=0

    y= [[],[],[],[],[]]

    x= [[],[],[],[],[]]

    for i in range(Nlabels):

        kmeans_plus=list(KMeans(n_clusters=clusters, random_state=0,max_iter=1000).fit(X_plus[i]).cluster_centers_) # mk+

        kmeans_neg= list(KMeans(n_clusters=clusters, random_state=0,max_iter=1000).fit(X_neg[i]).cluster_centers_) # mk-

        kmeans=kmeans_plus+kmeans_neg #mk+=mk-, mk=2mk+=2mk-

        #print(kmeans_neg)

        for k in range(Ndata):

            temp=[rand_x[k]]*2*clusters

            temp2=list(paired_distances(kmeans,temp,metric="euclidean")) # euclidean distance between datapoints and cluster centers

            temp2=list(scale(temp2))

            temp2=temp2+list(rand_x[k])



            x[i].append(temp2)

            y[i].append(rand_y[k][i])

  



    #print(len(x[0]),len(x[0][0]))

    for i in range(Nlabels):

        x[i]=np.array(x[i])

        y[i]=np.array(y[i])    



    X_train=np.array([None]*5)

    Y_train=np.array([None]*5)

    X_test=np.array([None]*5)

    Y_test=np.array([0]*5)

    classifier=[None]*5

    for i in range(0,5):

        classifier[i] = SVC(kernel='rbf', C=30,gamma='auto')



    pred=np.array([0]*5)

    loo=LeaveOneOut()

    acc_svm = []

    jac=[]





    for train, test in loo.split(x[0], y[0]):



        for i in range(Nlabels):

            X_train[i], X_test[i] = x[i][train], x[i][test]

            Y_train[i], Y_test[i] = y[i][train], y[i][test]

            #print(train)

            classifier[i].fit(X_train[i], Y_train[i])

            pred[i] = classifier[i].predict(X_test[i])



        

        

        if pred[4]==Y_test[4] and Y_test[4]==1:

            label_5=label_5+1

        elif pred[1]==Y_test[1] and Y_test[1]==1:

            label_2=label_2+1

        elif pred[2]==Y_test[2] and Y_test[2]==1:

            label_3=label_3+1

        elif pred[3]==Y_test[3] and Y_test[3]==1:

            label_4=label_4+1

        elif pred[0]==Y_test[0] and Y_test[0]==1:

            label_1=label_1+1

        

        prediction_sparse=lil_matrix(pred)

        labels_sparse=lil_matrix(Y_test)



        jac.append(jaccard_score(labels_sparse,prediction_sparse,average="samples"))

        acc_svm.append(accuracy_score(labels_sparse,prediction_sparse))







    

    print("for cluster=",clusters)

    print("label postives correctly predicted",label_1/X_plus[0].shape[0],"%",label_2/X_plus[1].shape[0],"%",label_3/X_plus[2].shape[0],"%",label_4/X_plus[3].shape[0],"%",label_5/X_plus[0].shape[0])

    print ('\n OAA: ',np.mean(acc_svm),np.std(acc_svm))

    print('\n  Acc',np.mean(jac),np.std(jac))