# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

# from main import method0



def data_clean(df, feature, m):

    count = df[feature].value_counts()

    df = df[df[feature].isin(count[count > m].index)]

    return df

def data_clean_sum(df,features,m):

    fil = df.ProductId.value_counts()

    fil2 = df.UserId.value_counts()

    df['#Proudcts'] = df.ProductId.apply(lambda x: fil[x])

    df['#Users'] = df.UserId.apply(lambda x: fil2[x])

    while (df.ProductId.value_counts(ascending=True)[0]) < m or  (df.UserId.value_counts(ascending=True)[0] < m):

        df = data_clean(df,features[0],m)

        df = data_clean(df,features[1],m)

    return df



# check if it is correct





def data():

    print('loading data...')

    df = pd.read_csv('../input/Reviews.csv')

    df['datetime'] = pd.to_datetime(df.Time, unit='s')

    raw_data = data_clean_sum(df, ['ProductId', 'UserId'], 10)

    # find X,and y

    raw_data['uid'] = pd.factorize(raw_data['UserId'])[0]

    raw_data['pid'] = pd.factorize(raw_data['ProductId'])[0]

    sc = MinMaxScaler()

    raw_data['time']=sc.fit_transform(raw_data['Time'].values.reshape(-1,1))

    raw_data['nuser']=sc.fit_transform(raw_data['#Users'].values.reshape(-1,1))

    raw_data['nproduct']=sc.fit_transform(raw_data['#Proudcts'].values.reshape(-1,1))

    # Sepreate the features into three groups

    X1 = raw_data.loc[:,['uid','pid']]

    X2 = raw_data.loc[:,['uid','pid','time']]

    X3 = raw_data.loc[:,['uid','pid','time','nuser','nproduct']]

    y = raw_data.Score

    # train_test split

    X1_train,X1_test,y_train,y_test = train_test_split(X1,y,test_size=0.3,random_state=2017)

    X2_train,X2_test,y_train,y_test = train_test_split(X2,y,test_size=0.3,random_state=2017)

    X3_train,X3_test,y_train,y_test = train_test_split(X3,y,test_size=0.3,random_state=2017)

    train = np.array(X1_train.join(y_train))

    test = np.array(X1_test.join(y_test))

    # got the productId to pid index

    pid2PID = raw_data.ProductId.unique()



    data_mixed = X1.join(y)

    total_p = data_mixed['pid'].unique().shape[0]

    total_u = data_mixed['uid'].unique().shape[0]

    # make the user-item table

    table = np.zeros([total_u,total_p])

    z = np.array(data_mixed)

    for line in z:

        u,p,s = line

        if table[u][p] < s:

            table[u][p] = s #if some one score a single thing several times

    print('the table\'s shape is:' )

    print(table.shape)

    return z, total_u,total_p,pid2PID,train,test,table,raw_data



z, total_u,total_p,pid2PID,train,test,table,raw_data = data()





from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def caculate_mse(x):

    MSE1=[]

    MSE2=[]

    for line in train:

        u,p,s = line

        MSE1.append(s)

        MSE2.append(x[u,p])

    MSE_in_sample = mean_squared_error(MSE1,MSE2)

    MSE3=[]

    MSE4 = []

    for line in test:

        u,p,s = line

        MSE3.append(s)

        MSE4.append(x[u,p])

    MSE_out_sample = mean_squared_error(MSE3,MSE4)

    print('the in sample MSE = {} \nthe out sample MSE = {}'.format(MSE_in_sample,MSE_out_sample))

    return MSE_in_sample,MSE_out_sample





def draw_mse(method,maxIter):

    import time

    c = []

    d = []

    timetime = []

    for i in [1,2,5,7,10,20,50,70,100]:

        tic = time.time()

        data = method(factors=i,maxIter=maxIter)

        a,b = caculate_mse(data)

        c.append(a)

        d.append(b)

        toc = time.time()

        timetime.append(toc-tic)

    aa = [1, 2, 5, 7, 10, 20, 50, 70, 100]

    for i in range(len(timetime)):



        print('latent factors = {}, time = {}'.format(aa[i],timetime[i]))

    plt.figure()

    plt.plot(aa,c,label = 'in_sample_MSE')

    plt.plot(aa,d,label = 'out_sample_MSE')

    plt.xticks([1,2,5,7,10,20,50,70,100])

    plt.legend()

    plt.show()

    return 0





import itertools

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')







def drawcm(y_pred,y_test =test ,title=''):

    print('caculating cm..')

    y1=[]

    y2=[]

    for line in y_test:

        u,p,s = line

        y1.append(s)

        y2.append(y_pred[u,p])

    temp1 = []

    temp2 = []

    for i in range(len(y1)):

        if np.array(y1)[i] >= 4:

            temp1.append(1)

        elif np.array(y1)[i] <= 2:

            temp1.append(0)

        else:

            temp1.append(0)

        if y2[i] >= 4:

            temp2.append(1)

        elif y2[i] <= 2:

            temp2.append(0)

        else:

            temp2.append(0)

    cm = confusion_matrix(temp1,temp2)

    plt.figure()

    plot_confusion_matrix(cm, classes=['not','recommand'], normalize=True,

                          title=title)

    plt.show()

from sklearn.metrics import *

from sklearn.preprocessing import *

from sklearn.ensemble import *

def rf():

    # find X,and y

    raw_data['uid'] = pd.factorize(raw_data['UserId'])[0]

    raw_data['pid'] = pd.factorize(raw_data['ProductId'])[0]

    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler()

    raw_data['time']=sc.fit_transform(raw_data['Time'].values.reshape(-1,1))

    raw_data['nuser']=sc.fit_transform(raw_data['#Users'].values.reshape(-1,1))

    raw_data['nproduct']=sc.fit_transform(raw_data['#Proudcts'].values.reshape(-1,1))



    X1 = raw_data.loc[:,['uid','pid']]

    X2 = raw_data.loc[:,['uid','pid','time']]

    X3 = raw_data.loc[:,['uid','pid','time','nuser','nproduct']]

    y = raw_data.Score



    from sklearn.model_selection import train_test_split

    X1_train,X1_test,y_train,y_test = train_test_split(X1,y,test_size=0.3,random_state=2017)

    X2_train,X2_test,y_train,y_test = train_test_split(X2,y,test_size=0.3,random_state=2017)

    X3_train,X3_test,y_train,y_test = train_test_split(X3,y,test_size=0.3,random_state=2017)

    a=RandomForestRegressor()

    a.fit(X3_train,y_train)

    y3 = a.predict(X3_test)

    sc = MinMaxScaler(feature_range=(1,5))

    c = mean_squared_error(y_train,a.predict(X3_train)), mean_squared_error(y_test,sc.fit_transform(y3.reshape(-1,1)))

    b = mean_squared_error(y_test,y3)

    print('train MSE is {}, test MSE is {}'.format(c,b))



    c3 = y3>=4

    t = y_test>=4

    print('accrucy of recommandtion:')

    print(accuracy_score(t,c3))

    c31 = y3<=1

    t1 = y_test<=1

    print('accrucy of not recommandtion:')

    print(accuracy_score(t1,c31))

    y_pred3 = []

    y_test3 = []

    for i in range(y3.shape[0]):

        if y3[i]>=4:

            y_pred3.append(1)

        elif y3[i]<4:

            y_pred3.append(0)

        # else:

            # y_pred3.append(1)



    for j in range(y3.shape[0]):

        if np.array(y_test)[j]>=4:

            y_test3.append(1)

        elif np.array(y_test)[j]<4:

            y_test3.append(0)

        # else:

            # y_test3.append(1)

    import itertools

    import matplotlib.pyplot as plt

    def plot_confusion_matrix(cm, classes,

                              normalize=False,

                              title='Confusion matrix',

                              cmap=plt.cm.Blues):

        """

        This function prints and plots the confusion matrix.

        Normalization can be applied by setting `normalize=True`.

        """

        if normalize:

            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            print("Normalized confusion matrix")

        else:

            print('Confusion matrix, without normalization')



        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.title(title)

        plt.colorbar()

        tick_marks = np.arange(len(classes))

        plt.xticks(tick_marks, classes, rotation=45)

        plt.yticks(tick_marks, classes)



        fmt = '.2f' if normalize else 'd'

        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            plt.text(j, i, format(cm[i, j], fmt),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")



        plt.tight_layout()

        plt.ylabel('True label')

        plt.xlabel('Predicted label')

    class_names = ['not recommand','recommand']

    cnf_matrix = confusion_matrix(y_test3,y_pred3)

    np.set_printoptions(precision=2)

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                          title='rf')





    plt.show()

    return a

rf()
def rec(result, uid,n,rawId= False):

    if uid in range(total_u):

		# we take the first n people's highest score product

        top_N = np.argpartition(result[uid],-n)[-n:]

        print('the top{} recommanded products for user {} is {}'.format(n,uid,top_N))

		# if rawID is on, the out put contains the real product id

        if rawId == True:

            print('the real ID is {}'.format(pid2PID[top_N]))

    else:

        print('this user has not bought anything, plz use other methods')

    return top_N


from sklearn.metrics.pairwise import pairwise_distances

def cf(table = table,distance = 'cosine'):

    user_similarity = pairwise_distances(table, metric=distance)

    item_similarity = pairwise_distances(table.T, metric=distance)

    sc = MinMaxScaler(feature_range=(1,5))

    a = sc.fit_transform(np.dot(user_similarity,table).dot(item_similarity))

    return a

result =cf()

caculate_mse(result)

drawcm(result,title='MF')

rec(result, 10,10,rawId= True)
from numpy import *

from scipy.sparse.linalg import svds

from numpy import linalg as la

from sklearn.preprocessing import MinMaxScaler

def svdrec(table = table, factors= 150):

    UI = matrix(table)

    # ui_df = pd.DataFrame(UI,index=table.index, columns=table.columns)

    user_ratings_mean=mean(UI,axis=0)

    user_ratings_mean=user_ratings_mean.reshape(1,-1)

    UI_demeaned=UI-user_ratings_mean

    U,sigma,Vt=svds(UI_demeaned,factors)

    sigma=diag(sigma)

    pred_mat=dot(dot(U,sigma),Vt) + user_ratings_mean

    sc=MinMaxScaler(feature_range = (1,5))

    pred_mat = sc.fit_transform(pred_mat)

    # prediction_df=pd.DataFrame(pred_mat,index=table.index,columns=table.columns)

    return pred_mat

def rec(result, uid,n,rawId= False):

    if uid in range(total_u):

		# we take the first n people's highest score product

        top_N = np.argpartition(result[uid],-n)[-n:]

        print('the top{} recommanded products for user {} is {}'.format(n,uid,top_N))

		# if rawID is on, the out put contains the real product id

        if rawId == True:

            print('the real ID is {}'.format(pid2PID[top_N]))

    else:

        print('this user has not bought anything, plz use other methods')

    return top_N

result1 =svdrec(factors=150)

caculate_mse(result1)

drawcm(result1,title='SVD')

rec(result1, 10,10,rawId= True)
def MF1(data=z, factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, plot=False):

    # initial the latent matrix for user and item

    P = np.random.rand(total_u, factors) / 3

    Q = np.random.rand(total_p, factors) / 3

    # initial y as the history of loss

    y = []

    # initial the iteration and last loss

    iteration = 0

    last_loss = 0

    while iteration < maxIter:

        loss = 0

        for i in range(data.shape[0]):

            # get the uid,pid and the score from every line

            u, p, s = data[i]

            # calculate the error

            error = s - np.dot(P[u], Q[p])

            # calculate the loss function

            # avoid loss become to large, scale to 1/50

            loss += error ** 2 / 50

            # update the parameter according to the gradient descent

            pp = P[u]

            qq = Q[p]

            P[u] += LRate * error * qq

            Q[p] += LRate * error * pp

        iteration += 1

        y.append(loss)

        delta_loss = last_loss - loss

        print('iter = {}, loss = {}, delta_loss = {}, LR = {}'.format(iteration, loss, delta_loss, LRate))

        # update the learn rate to make sure it will converge

        if abs(last_loss) > abs(loss):

            LRate *= 1.05

        else:

            LRate *= 0.5

        # When converge, stop the gradient descend

        if abs(delta_loss) < abs(GD_end):

            print('the diff in loss is {}, so the GD stops'.format(delta_loss))

            break

        last_loss = loss

    if plot:

        plt.plot(y)

        plt.show()

    return P.dot(Q.T)



result =MF1( factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, plot=1)

caculate_mse(result)

drawcm(result,title='MF')

def rec(result, uid,n,rawId= False):

    if uid in range(total_u):

		# we take the first n people's highest score product

        top_N = np.argpartition(result[uid],-n)[-n:]

        print('the top{} recommanded products for user {} is {}'.format(n,uid,top_N))

		# if rawID is on, the out put contains the real product id

        if rawId == True:

            print('the real ID is {}'.format(pid2PID[top_N]))

    else:

        print('this user has not bought anything, plz use other methods')

    return top_N

rec(result, 10,10,rawId= True)
def PMF(data=z, factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, regU = 0.01 ,regI = 0.01 ,plot=False):

    P = np.random.rand(total_u, factors) / 3

    Q = np.random.rand(total_p, factors) / 3

    y = []

    iteration = 0

    last_loss = 100

    while iteration < maxIter:

        loss = 0

        for i in range(data.shape[0]):

            u, p, s = data[i]

            error = s - np.dot(P[u], Q[p])

            loss += error ** 2/50

            pp = P[u]

            qq = Q[p]

            P[u] += LRate *  (error * qq - regU*pp)

            Q[p] += LRate * (error * pp - regI * qq)

        loss += regU*(P*P).sum() +regI*(Q*Q).sum()

        iteration += 1

        y.append(loss)

        delta_loss = last_loss - loss

        print('iter = {}, loss = {}, delta_loss = {}, LR = {}'.format(iteration, loss, delta_loss, LRate))

        if abs(last_loss) > abs(loss):

            LRate *= 1.05

        else:

            LRate *= 0.5



        if abs(delta_loss) < abs(GD_end):

            print('the diff in loss is {}, so the GD stops'.format(delta_loss))

            break

        last_loss = loss

    if plot:

        plt.plot(y)

        plt.show()

    return P.dot(Q.T)

result =PMF( factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, plot=1)

caculate_mse(result)

drawcm(result,title='PMF')

def rec(result, uid,n,rawId= False):

    if uid in range(total_u):

		# we take the first n people's highest score product

        top_N = np.argpartition(result[uid],-n)[-n:]

        print('the top{} recommanded products for user {} is {}'.format(n,uid,top_N))

		# if rawID is on, the out put contains the real product id

        if rawId == True:

            print('the real ID is {}'.format(pid2PID[top_N]))

    else:

        print('this user has not bought anything, plz use other methods')

    return top_N

rec(result, 10,10,rawId= True)