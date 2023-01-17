import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



ddx=pd.read_csv('../input/symptom-disease-dataset-2/syditriage.csv')

# Any results you write to the current directory are saved as output.
ddx.columns=['sym','symptom','dis','diagnose','dg','wei']

ddx
import scipy.sparse as sparse



cleaned_retail = ddx[['wei', 'symptom', 'diagnose']] # Get rid of unnecessary info



grouped_cleaned = cleaned_retail.groupby(['diagnose', 'symptom']).sum().reset_index() # Group together

grouped_cleaned.wei.loc[grouped_cleaned.wei == 0] = 1 # Replace a sum of zero purchases with a one to

# indicate purchased

grouped_purchased = grouped_cleaned.query('wei > 50') # Only get customers where purchase totals were positive

print(grouped_purchased.info())

customers = list(np.sort(grouped_purchased.symptom.unique())) # Get our unique customers

products = list(grouped_purchased.diagnose.unique()) # Get our unique products that were purchased

quantity = list(grouped_purchased.wei) # All of our purchases



rows = grouped_purchased.symptom.astype('category', categorie = customers).cat.codes 

# Get the associated row indices

cols = grouped_purchased.diagnose.astype('category', categorie = products).cat.codes 

# Get the associated column indices

purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))
ddx.groupby(['diagnose']).mean(),pd.DataFrame(customers)
basket_sets = pd.pivot_table(ddx, index='symptom',columns='diagnose',values='wei')

basket_sets.info()

#atc can be linked to atc, i simplify here

item_lookup = pd.DataFrame( list( basket_sets.columns ) ,columns=['StockCode'])# Only get unique item/description pairs

#item_lookup=pd.merge(item_lookup, atc, how='left', on=None, left_on='StockCode', right_on='ATC')

#item_lookup.columns=['StockCode','kegg','Description','ATC','CAS','PubChem','Snomed']

item_lookup.head()
matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1] # Number of possible interactions in the matrix

num_purchases = len(purchases_sparse.nonzero()[0]) # Number of items interacted with

sparsity = 100*(1 - (num_purchases/matrix_size))

sparsity

def make_train(ratings, pct_test = 0.2):

    import random

    '''

    This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a

    user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings, 

    while the training set replaces the specified percentage of them with a zero in the original ratings matrix. 

    

    parameters: 

    

    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete

    copy of the original set. This is in the form of a sparse csr_matrix. 

    

    pct_test - The percentage of user-item interactions where an interaction took place that you want to mask in the 

    training set for later comparison to the test set, which contains all of the original ratings. 

    

    returns:

    

    training_set - The altered version of the original data with a certain percentage of the user-item pairs 

    that originally had interaction set back to zero.

    

    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order 

    compares with the actual interactions.

    

    user_inds - From the randomly selected user-item indices, which user rows were altered in the training data.

    This will be necessary later when evaluating the performance via AUC.

    '''

    test_set = ratings.copy() # Make a copy of the original set to be the test set. 

    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix

    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 

    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists

    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list

    random.seed(0) # Set the random seed to zero for reproducibility

    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer

    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement

    user_inds = [index[0] for index in samples] # Get the user row indices

    item_inds = [index[1] for index in samples] # Get the item column indices

    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero

    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space

    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered  
product_train, product_test, product_users_altered = make_train(purchases_sparse, pct_test = 0.01)

product_train
def implicit_weighted_ALS(training_set, lambda_val = 0.1, alpha = 40, iterations = 10, rank_size = 20, seed = 0):

    '''

    Implicit weighted ALS taken from Hu, Koren, and Volinsky 2008. Designed for alternating least squares and implicit

    feedback based collaborative filtering. 

    

    parameters:

    

    training_set - Our matrix of ratings with shape m x n, where m is the number of users and n is the number of items.

    Should be a sparse csr matrix to save space. 

    

    lambda_val - Used for regularization during alternating least squares. Increasing this value may increase bias

    but decrease variance. Default is 0.1. 

    

    alpha - The parameter associated with the confidence matrix discussed in the paper, where Cui = 1 + alpha*Rui. 

    The paper found a default of 40 most effective. Decreasing this will decrease the variability in confidence between

    various ratings.

    

    iterations - The number of times to alternate between both user feature vector and item feature vector in

    alternating least squares. More iterations will allow better convergence at the cost of increased computation. 

    The authors found 10 iterations was sufficient, but more may be required to converge. 

    

    rank_size - The number of latent features in the user/item feature vectors. The paper recommends varying this 

    between 20-200. Increasing the number of features may overfit but could reduce bias. 

    

    seed - Set the seed for reproducible results

    

    returns:

    

    The feature vectors for users and items. The dot product of these feature vectors should give you the expected 

    "rating" at each point in your original matrix. 

    '''

    

    # first set up our confidence matrix

    

    conf = (alpha*training_set) # To allow the matrix to stay sparse, I will add one later when each row is taken 

                                # and converted to dense. 

    num_user = conf.shape[0]

    num_item = conf.shape[1] # Get the size of our original ratings matrix, m x n

    

    # initialize our X/Y feature vectors randomly with a set seed

    rstate = np.random.RandomState(seed)

    

    X = sparse.csr_matrix(rstate.normal(size = (num_user, rank_size))) # Random numbers in a m x rank shape

    Y = sparse.csr_matrix(rstate.normal(size = (num_item, rank_size))) # Normally this would be rank x n but we can 

                                                                 # transpose at the end. Makes calculation more simple.

    X_eye = sparse.eye(num_user)

    Y_eye = sparse.eye(num_item)

    lambda_eye = lambda_val * sparse.eye(rank_size) # Our regularization term lambda*I. 

    

    # We can compute this before iteration starts. 

    

    # Begin iterations

   

    for iter_step in range(iterations): # Iterate back and forth between solving X given fixed Y and vice versa

        # Compute yTy and xTx at beginning of each iteration to save computing time

        yTy = Y.T.dot(Y)

        xTx = X.T.dot(X)

        # Being iteration to solve for X based on fixed Y

        for u in range(num_user):

            conf_samp = conf[u,:].toarray() # Grab user row from confidence matrix and convert to dense

            pref = conf_samp.copy() 

            pref[pref != 0] = 1 # Create binarized preference vector 

            CuI = sparse.diags(conf_samp, [0]) # Get Cu - I term, don't need to subtract 1 since we never added it 

            yTCuIY = Y.T.dot(CuI).dot(Y) # This is the yT(Cu-I)Y term 

            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T) # This is the yTCuPu term, where we add the eye back in

                                                      # Cu - I + I = Cu

            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu) 

            # Solve for Xu = ((yTy + yT(Cu-I)Y + lambda*I)^-1)yTCuPu, equation 4 from the paper  

        # Begin iteration to solve for Y based on fixed X 

        for i in range(num_item):

            conf_samp = conf[:,i].T.toarray() # transpose to get it in row format and convert to dense

            pref = conf_samp.copy()

            pref[pref != 0] = 1 # Create binarized preference vector

            CiI = sparse.diags(conf_samp, [0]) # Get Ci - I term, don't need to subtract 1 since we never added it

            xTCiIX = X.T.dot(CiI).dot(X) # This is the xT(Cu-I)X term

            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T) # This is the xTCiPi term

            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)

            # Solve for Yi = ((xTx + xT(Cu-I)X) + lambda*I)^-1)xTCiPi, equation 5 from the paper

    # End iterations

    return X, Y.T # Transpose at the end to make up for not being transposed at the beginning. 

                         # Y needs to be rank x n. Keep these as separate matrices for scale reasons. 

#TOO SLOW

#user_vecs, item_vecs = #implicit_weighted_ALS(product_train, lambda_val = 0.1, alpha = 15, iterations = 1,rank_size = 20)

#user_vecs
import implicit

alpha = 15

user_vecs, item_vecs = implicit.alternating_least_squares(product_train.astype('double'), 

                                                          factors=150, 

                                                          regularization = 0.1, 

                                                         iterations = 50)
from scipy.sparse.linalg import svds, eigs

#user_vecs, s, item_vecs = svds(product_train.astype('double'), k=150)  #10 0.38 >40=0.31 will it jump higher now

#item_vecs=item_vecs.T



#svd is worse 62% AUC
#user_vecs=cluster_embedding

#item_vecs=cluster_embedding_symp

#umap is worse 50% AUC
user_vecs[0,:].dot(item_vecs.T)[:5]

import matplotlib.pyplot as plt

# synthetic reconstructed weighted disease-symptom matrix

plt.imshow(user_vecs.dot(item_vecs.T))

from sklearn import metrics

def auc_score(predictions, test):

    '''

    This simple function will output the area under the curve using sklearn's metrics. 

    

    parameters:

    

    - predictions: your prediction output

    

    - test: the actual target result you are comparing to

    

    returns:

    

    - AUC (area under the Receiver Operating Characterisic curve)

    '''

    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)

    return metrics.auc(fpr, tpr) 



def calc_mean_auc(training_set, altered_users, predictions, test_set):

    '''

    This function will calculate the mean AUC by user for any user that had their user-item matrix altered. 

    

    parameters:

    

    training_set - The training set resulting from make_train, where a certain percentage of the original

    user/item interactions are reset to zero to hide them from the model 

    

    predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.

    These should be stored in a list, with user vectors as item zero and item vectors as item one. 

    

    altered_users - The indices of the users where at least one user/item pair was altered from make_train function

    

    test_set - The test set constucted earlier from make_train function

    

    

    

    returns:

    

    The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions

    there were originally zero to test ranking ability in addition to the most popular items as a benchmark.

    '''

    

    

    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set

    popularity_auc = [] # To store popular AUC scores

    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular

    item_vecs = predictions[1]

    for user in altered_users: # Iterate through each user that had an item altered

        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row

        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred

        # Get the predicted values based on our user/item vectors

        user_vec = predictions[0][user,:]

        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)

        # Get only the items that were originally zero

        # Select all ratings from the MF prediction for this user that originally had no iteraction

        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 

        # Select the binarized yes/no interaction pairs from the original full data

        # that align with the same pairs in training 

        pop = pop_items[zero_inds] # Get the item popularity for our chosen items

        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store

        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score

    # End users iteration

    

    return float('%.3f'%np.mean(store_auc)),' > is better < worse ' ,float('%.3f'%np.mean(popularity_auc))  

   # Return the mean AUC rounded to three decimal places for both test and popularity benchmark

    

calc_mean_auc(product_train, product_users_altered, 

              [sparse.csr_matrix(user_vecs), sparse.csr_matrix(item_vecs.T)], product_test)

# AUC for our recommender system
customers_arr = np.array(customers) # Array of customer IDs from the ratings matrix

products_arr = np.array(products) # Array of product IDs from the ratings matrix
def get_items_purchased(customer_id, mf_train, customers_list, products_list, item_lookup):

    '''

    This just tells me which items have been already purchased by a specific user in the training set. 

    

    parameters: 

    

    customer_id - Input the customer's id number that you want to see prior purchases of at least once

    

    mf_train - The initial ratings training set used (without weights applied)

    

    customers_list - The array of customers used in the ratings matrix

    

    products_list - The array of products used in the ratings matrix

    

    item_lookup - A simple pandas dataframe of the unique product ID/product descriptions available

    

    returns:

    

    A list of item IDs and item descriptions for a particular customer that were already purchased in the training set

    '''

    cust_ind = np.where(customers_list == customer_id)[0][0] # Returns the index row of our customer id

    purchased_ind = mf_train[cust_ind,:].nonzero()[1] # Get column indices of purchased items

    prod_codes = products_list[purchased_ind] # Get the stock codes for our purchased items

    return item_lookup.loc[item_lookup.StockCode.isin(prod_codes)]
customers_arr
get_items_purchased(customers_arr[-1:], product_train, customers_arr, products_arr, item_lookup),customers_arr[-1:]
from sklearn.preprocessing import MinMaxScaler



def rec_items(customer_id, mf_train, user_vecs, item_vecs, customer_list, item_list, item_lookup, num_items = 10):

    '''

    This function will return the top recommended items to our users 

    

    parameters:

    

    customer_id - Input the customer's id number that you want to get recommendations for

    

    mf_train - The training matrix you used for matrix factorization fitting

    

    user_vecs - the user vectors from your fitted matrix factorization

    

    item_vecs - the item vectors from your fitted matrix factorization

    

    customer_list - an array of the customer's ID numbers that make up the rows of your ratings matrix 

                    (in order of matrix)

    

    item_list - an array of the products that make up the columns of your ratings matrix

                    (in order of matrix)

    

    item_lookup - A simple pandas dataframe of the unique product ID/product descriptions available

    

    num_items - The number of items you want to recommend in order of best recommendations. Default is 10. 

    

    returns:

    

    - The top n recommendations chosen based on the user/item vectors for items never interacted with/purchased

    '''

    

    cust_ind = np.where(customer_list == customer_id)[0][0] # Returns the index row of our customer id

    pref_vec = mf_train[cust_ind,:].toarray() # Get the ratings from the training set ratings matrix

    pref_vec = pref_vec.reshape(-1) + 1 # Add 1 to everything, so that items not purchased yet become equal to 1

    print(pref_vec)

    pref_vec[pref_vec > 2] = 0 # Make everything already purchased zero

    rec_vector = user_vecs[cust_ind,:].dot(item_vecs.T) # Get dot product of user vector and all item vectors

    # Scale this recommendation vector between 0 and 1

    min_max = MinMaxScaler()

    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0] 

    recommend_vector = rec_vector_scaled #*pref_vec

    # Items already purchased have their recommendation multiplied by zero

    product_idx = np.argsort(recommend_vector)[::-1][:num_items] # Sort the indices of the items into order 

    

    # of best recommendations

    rec_list = [] # start empty list to store items

    for index in product_idx:

        code = item_list[index]

        rec_list.append([code, item_lookup.StockCode.loc[item_lookup.StockCode == code].iloc[0]]) 

        # Append our descriptions to the list

    codes = [item[0] for item in rec_list]

    descriptions = [item[1] for item in rec_list]

    print(codes,descriptions)

    final_frame = pd.DataFrame({'StockCode': codes, 'Description': descriptions}) # Create a dataframe 

    return final_frame[['StockCode', 'Description']] # Switch order of columns around

item_lookup
rec_items(customers_arr[882], product_train, user_vecs, item_vecs, customers_arr, products_arr, item_lookup,num_items = 20),get_items_purchased(customers_arr[882], product_train, customers_arr, products_arr, item_lookup),customers_arr[882]
#fe typical problem Gastritis

ddx[ddx.diagnose=='Allergic rhinitis']
#transformed here

item_lookup[item_lookup.StockCode=='Allergic rhinitis']
#user/customer=symptoms = rows 19133

#item/product=diseases = columns 8128

user_vecs.shape,item_vecs.shape,item_lookup.shape,product_train.shape, customers_arr.shape, products_arr.shape,product_test.shape, 
[i for i,x in enumerate(customers_arr) if x=='Allergic rhinitis']

ddx.groupby('sym').max()
#user/customer=symptoms = rows 19133

#item/product=diseases = columns 8128

# search a disease relevant questions !



zoekziekte=[i for i,x in enumerate(products_arr) if x=='Sinusitis'][0]

print(zoekziekte)

impsymp=pd.DataFrame()

impsymp['rating']=user_vecs.dot(item_vecs[zoekziekte].T)

impsymp['symptoms']=customers_arr

impsymp.sort_values('rating',ascending=False)[:25]
#compare

ddx[ddx.diagnose=='Sinusitis'].sort_values('sym')
ddx[ddx.symptom=='Runny nose'].sort_values('wei',ascending=False)[:10]
ddx[   (ddx.symptom=='Sneezing') | (ddx.symptom=='Runny nose')   ].sort_values('wei',ascending=False)[:20]
#one symptom relevant diseases ?



zoeksymp=[i for i,x in enumerate(customers_arr) if x=='Runny nose'][0]

print(zoeksymp)

impdis=pd.DataFrame()

impdis['rating']=user_vecs[zoeksymp].dot(item_vecs.T)

impdis['disease']=products_arr

impdis.sort_values('rating',ascending=False)[:20]
[x for x in customers_arr if x in ['Sneezing','Runny nose']  ]
#one symptom relevant diseases ?

searchsymp=['Sneezing','Runny nose']

zoeksymp=[i for i,x in enumerate(customers_arr) if x in searchsymp]

print(zoeksymp)



impdis2=pd.DataFrame()

impdis=user_vecs[zoeksymp].dot(item_vecs.T)

for xi,si in enumerate(searchsymp):

    impdis2[searchsymp[xi]]=impdis[xi,:]

impdis2['rating']=impdis2[searchsymp[0]]*impdis2[searchsymp[1]]

impdis2['disease']=products_arr

impdis2.sort_values('rating',ascending=False)[:20]
#compare

searchsymp=ddx[ddx.diagnose=='Allergic rhinitis'].symptom.values

zoeksymp=[i for i,x in enumerate(customers_arr) if x in searchsymp]

print(zoeksymp)



zoeksymp=[i for i,x in enumerate(customers_arr) if x=='Rhinorrhea'][0]

print(zoeksymp)
#search one symptom most similar symptoms

zoeksymp=[i for i,x in enumerate(customers_arr) if x=='Runny nose'][0]

print(zoeksymp)

user_vecs[zoeksymp]



from sklearn.metrics.pairwise import cosine_similarity



bestsymp=cosine_similarity(user_vecs,[user_vecs[zoeksymp]])

print(bestsymp.T[0])

impsymp=pd.DataFrame()

impsymp['rating']=bestsymp.T[0]

impsymp['symptom']=customers_arr

impsymp.sort_values('rating',ascending=False)[:20]
zoeksymp=[i for i,x in enumerate(customers_arr) if x in ['Runny nose','Sneezing','Itching eyes']]

notsymp=[i for i,x in enumerate(customers_arr) if x in ['Headaches']]



print(zoeksymp,notsymp)

user_vecs[zoeksymp]



from sklearn.metrics.pairwise import cosine_similarity



bestvect=cosine_similarity(user_vecs,user_vecs[zoeksymp])

notvect=cosine_similarity(user_vecs,user_vecs[notsymp])



impsymp=pd.DataFrame()

impsymp['rating']=-notvect.T[0]

for xi in range(len(zoeksymp)):

    impsymp['rating']+=bestvect.T[xi]

impsymp['symptom']=customers_arr

impsymp.sort_values('rating',ascending=False)[:20]
impdis2=pd.DataFrame()

impdis=user_vecs[zoeksymp+notsymp].dot(item_vecs.T)

for xi,si in enumerate(zoeksymp):

    impdis2[zoeksymp[xi]]=impdis[xi,:]

impdis2[notsymp[0]]=impdis[3,:]

impdis2['rating']=impdis2[zoeksymp[0]]+impdis2[zoeksymp[1]]+impdis2[zoeksymp[2]]-impdis2[notsymp[0]]

impdis2['disease']=products_arr

impdis2.sort_values('rating',ascending=False)[:20]
impdis3=impdis2.sort_values('rating',ascending=False)[:20].merge(ddx,how='left',left_on='disease',right_on='diagnose').groupby('dis').max()

impdis3['rate2']=impdis3['rating']*impdis3['wei']

impdis3.sort_values('rate2')
#user/customer=symptoms = rows 19133

#item/product=diseases = columns 8128

# search a disease relevant questions !





impdis4=pd.DataFrame()

temp= (user_vecs[zoeksymp+notsymp].dot(item_vecs.T)).sum(axis=0)

product_train.shape,temp.shape



impdis4=pd.DataFrame(cosine_similarity(user_vecs[zoeksymp],item_vecs)  ).T

impdis4['disease']=products_arr

impdis4['rating']=impdis4[0]+impdis4[1]+impdis4[2]

impdis5=impdis4.sort_values('rating',ascending=False)[:100].merge(ddx,how='left',left_on='disease',right_on='diagnose').groupby('dis').max()

impdis5.sort_values('rating',ascending=False)[:20]
#top10 symptoms

def top10symp(symptoom_oms,symptoomnr,aantal):

    #search one symptom most similar symptoms

    #zoeksymp=[i for i,x in enumerate(customers_arr) if x==symptoom_oms][0]

    #print(zoeksymp)

    user_vecs[symptoomnr]



    from sklearn.metrics.pairwise import cosine_similarity



    bestsymp=cosine_similarity(user_vecs,[user_vecs[symptoomnr]])

    impsymp=pd.DataFrame()

    

    impsymp['rank']=bestsymp.T[0]

    impsymp['uppersymptom']=symptoomnr

    impsymp['upperoms']=symptoom_oms

    impsymp['symptom']=customers_arr

    return impsymp.sort_values('rank',ascending=False)[:aantal+1]

    

top10symp('Runny nose or nasal congestion',8400,10)
#top diseases per two symptompairs

def top10disease(zoeksymp,aantal):

    impdis4=pd.DataFrame()

    temp= (user_vecs[zoeksymp].dot(item_vecs.T)).sum(axis=0)

    impdis4=pd.DataFrame(cosine_similarity(user_vecs[zoeksymp],item_vecs)  ).T

    impdis4['disease']=products_arr

    impdis4['rating']=impdis4[0]+impdis4[1]

    impdis4['sym1']=zoeksymp[0]

    impdis4['sym2']=zoeksymp[1]

    impdis5=impdis4.sort_values('rating',ascending=False)[:100].merge(ddx,how='left',left_on='disease',right_on='diagnose').groupby('dis').max()

    return impdis5.sort_values('rating',ascending=False)[:aantal]



top10disease([0,2227],10)
symptunique=ddx.groupby(['sym','symptom']).max()

symptunique=symptunique[:300]

symptunique
tree=top10symp('',0,10)

for xi,ci in enumerate(symptunique.reset_index().symptom):

    print(xi,ci)

    tree=tree.append(top10symp(ci,xi,10))

    

tree
tree.index[30]