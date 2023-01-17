import os



import time

from numpy.linalg import matrix_rank

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



dir='../input/movielens-100k/u_100k.data'





class MF(object):



    def __init__(self, R, K, alpha, beta, iterations):

        """

        Perform matrix factorization to predict empty

        entries in a matrix.



        Arguments

        - R (ndarray)   : user-item rating matrix

        - K (int)       : number of latent dimensions

        - alpha (float) : learning rate

        - beta (float)  : regularization parameter

        """



        self.R = R

        self.num_users, self.num_items = R.shape

        self.K = K

        self.alpha = alpha

        self.beta = beta

        self.iterations = iterations



    def train(self):

        # Initialize user and item latent feature matrice

        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))

        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))



        # Initialize the biases

        self.b_u = np.zeros(self.num_users)

        self.b_i = np.zeros(self.num_items)

        self.b = np.mean(self.R[np.where(self.R != 0)])



        # Create a list of training samples

        self.samples = [

            (i, j, self.R[i, j])

            for i in range(self.num_users)

            for j in range(self.num_items)

            if self.R[i, j] > 0

        ]



        # Perform stochastic gradient descent for number of iterations

        training_process = []

        for i in range(self.iterations):

            np.random.shuffle(self.samples)

            self.sgd()

            mse = self.mse()

            

            #print("Iteration: %d ; error = %.4f" % (i+1, mse))

            training_process.append(int(round(mse)))

        return (training_process)



    def mse(self):

        """

        A function to compute the total mean square error

        """

        xs, ys = self.R.nonzero()

        

        predicted = self.full_matrix()

        error = 0

        for x, y in zip(xs, ys):

            error += pow(self.R[x, y] - predicted[x, y], 2)

        return np.sqrt(error)



    def sgd(self):

        """

        Perform stochastic graident descent

        """

        for i, j, r in self.samples:

            # Computer prediction and error

            prediction = self.get_rating(i, j)

            e = (r - prediction)



            # Update biases

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])

            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])



            # Update user and item latent feature matrices

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])

            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])



    def get_rating(self, i, j):

        """

        Get the predicted rating of user i and item j

        """

        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)

        return prediction



    def full_matrix(self):

        """

        Computer the full matrix using the resultant biases, P and Q

        """

        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

#We can try to apply it to our example mentioned above and see what we would get. Below is a code snippet in Python for running the example.







class Code():



    

    # Uploading Movie lens dataset

    header = ['user_id', 'item_id', 'rating', 'timestamp']

    df =pd.read_csv(dir,sep='\t')

    df.columns=header





    # Representing dataset as  User-Item matrix

    n_users = df.user_id.unique().shape[0]

    n_items = df.item_id.unique().shape[0]

    ratings = np.zeros((n_users, n_items))

    for row in df.itertuples():

        ratings[row[1]-1, row[2]-1] = row[3]



    #Finding non-zero values of the User-Item matrix

    non_zero = np.count_nonzero(ratings) 

          

    total_val = np.product(ratings.shape)



    #calculating sparsity level

    sparsity = round((total_val - non_zero) / total_val,3)

              

    #calculating sparsity percentage

    #print("The Sparsity level of Movielens 100k is",str(sparsity*100),"%")      



    #Dividing data into tarin and test set

    train_data, test_data = train_test_split(df, test_size=0.25)



    #Create two user-item matrices, one for training and another for testing

    train_data_matrix = np.zeros((n_users, n_items))

    

    #finding Rank for Train_data_matrix

    train_rank=matrix_rank(train_data_matrix)



    for line in train_data.itertuples():

        train_data_matrix[line[1]-1, line[2]-1] = line[3]



    test_data_matrix = np.zeros((n_users, n_items))

    for line in test_data.itertuples():

        test_data_matrix[line[1]-1, line[2]-1] = line[3]



if __name__=='__main__':



    #create an Empty DataFrame

    COLUMN_NAMES=['Sparsity','Epoches','Learning_Rate','Train_RMSE','Test_RMSE','Time_Taken'] 

    analysis = pd.DataFrame(columns=COLUMN_NAMES)

    

    pd.set_option('display.max_rows', None)

    pd.set_option('display.max_columns', None)

    pd.set_option('display.width', None)

    pd.set_option('display.max_colwidth', -1)

    

    #print(analysis)

        

    epochs=[]

    learning_rate=[]

    

    #Testing with different parameters

    print('Hello...... Enter No of TestCases')

    count=int(input())

    for i in range(count):

        print('Hello.....Enter the Epoches Value '+str(i)+': ')

        val=int(input())

        epochs.append(val)

        print('Hello.....Enter the Learning_Rates '+str(i)+': ')

        val=float(input())

        learning_rate.append(val)

          

    for i in range(0,count):

        

        obj=Code()



        #To store the Initial Sparsity into the DataFrame called Analysis

        sparse=str(obj.sparsity*100)+'%'



        #Reading the input of No of epoches and Learning_Rate and at the same time used to store the data into the DataFrame Called Analysis

        

        # To find the starting time of the execution

        #import MatrixFactorization as m

        start=time.time()

        #print(start)

        print('Training the data......')

        mf = MF(obj.train_data_matrix, K=obj.train_rank, alpha=learning_rate[i], beta=0.01, iterations=epochs[i])

        training_process = mf.train()

        end=time.time()



        #printing time taken for exection in seconds

        m,s=divmod(end-start,60)

        #print("time taken for training:",m,"minutes and",s,"Seconds" )

        #print()



        #To Store Time_taken into the DataFrame called Analysis

        time_taken=str(m)+" min "+str(int(s))+" Sec"



        #print("P x Q:")

        #print(mf.full_matrix())

        full_matrix=mf.full_matrix()

        non_zero = np.count_nonzero(full_matrix) 

        #print(non_zero)     

        total_val = np.product(full_matrix.shape)

        #print(total_val)

        sparsity = round(((total_val - non_zero) / total_val),3)



        #Calculating sparsity of reconstructed matrix

        #print("The Sparsity level of Movielens 100k is",str(sparsity*100),"%") 





        #Train the data using the test dataset

        #import MatrixFactorization as m

        mf = MF(obj.test_data_matrix, K=obj.train_rank, alpha=learning_rate[i], beta=0.01, iterations=epochs[i])

        test_process = mf.train()

        test_matrix=mf.full_matrix()





        from sklearn.metrics import mean_squared_error

        from math import sqrt

        def rmse(prediction, ground_truth):

                        prediction = prediction[ground_truth.nonzero()].flatten() 

                        ground_truth = ground_truth[ground_truth.nonzero()].flatten()

                        return sqrt(mean_squared_error(prediction, ground_truth))

         

        #print('RMSE: ',str(rmse(full_matrix, obj.test_data_matrix)))

        #Storing the Rmse in the Error variable to append the data into the DataFrame caleed Analysis

        error=rmse(test_matrix,obj.test_data_matrix)

        train_rmse=rmse(full_matrix,obj.train_data_matrix)



    

        analysis.loc[str(i)] = [sparse,epochs[i],learning_rate[i],train_rmse,error,time_taken]

    

    print(analysis)
