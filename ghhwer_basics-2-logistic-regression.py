import numpy as np # what im i supposed to say?

import pandas as pd # data processing

pd.set_option('mode.chained_assignment', None) # Nasty, i know...



import torch # To light the way

import matplotlib.pyplot as plt # xes and ys



from copy import deepcopy # Memory Things
def expands_variables(df, cats, conts, aggs=['mean', 'std', 'max', 'min'], powers=4, put_dummies=True):

    """

    This functions expands categorical fields by aggregating them and appending data as new rows

      this method also puts raises the continuous variables to powers if powers is not None

      It also creates dummies if you ask it to!

    """

    use_powers = (not powers==None) # Should use powers?

    groups = cats + list(set([tuple(sorted([x,y]))  for x in cats for y in cats if x!=y ])) # Permutes categorical to create combinations 

    cont_filters = {key:aggs for key in conts} # Agg dict

    temp = df.copy() # Create a copy from DataFrame

    

    # For each group do groupby and merge DataFrame

    for idx in range(len(groups)):

        g = groups[idx]

        # To get rid of annoying warnnings 

        if type(g) == tuple:

            g = list(g)

        gb = temp.groupby(g).agg(cont_filters) # GroupBy

        gb.columns = ["_".join([x[0],x[1],str(idx)]) for x in list(gb.columns)] # Rename columns so they don't overlap

        temp = pd.merge(temp, gb, on=g, how="left") # Merge DataFrame

    

    if use_powers: # If you desire to use powers

        for x in conts:

            for pw in range(powers-1):

                temp[x+'_pow_'+str(pw+2)] = temp[x]**(pw+2) # Raise them powers and put to their respective names

    if put_dummies:# If you desire to use dummies

        for x in cats: 

            temp = temp.join(pd.get_dummies(temp[x], dtype=float), how='left') # Get them dummies

    return temp # return transformed DataFrame
dataset = pd.read_csv('../input/diabetes-dataset/diabetes2.csv') # Load

dataset = expands_variables(dataset, [], dataset.drop(['Outcome'], axis=1).columns) # Sprinkle Some Info

dataset = dataset.sample(dataset.count()[0]) # Randomize



#Split

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(dataset, test_size=0.2)
def normalize_data(data, ignore=None):

    df = deepcopy(data)

    describe = dataset.describe()

    mapping = {}

    for x in df.columns:

        if x not in ignore:

            desc = dataset.describe().T[['mean','std']]

            mean = desc.loc[desc.index==x]['mean'].values[0]

            std = desc.loc[desc.index==x]['std'].values[0]

            mapping[x] = {'mean':mean,'std':std}

            df[x] = df[x].apply(lambda x: (x-mean)/std)

    return df, mapping



def denormalize_data(data, mapping):

    df = deepcopy(data)

    for x in mapping.keys():

        df[x] = df[x].apply(lambda u: (u*mapping[x]['std'])+mapping[x]['mean'])

    return df, mapping
train_df_norm, normal_mapping = normalize_data(train_df, ignore=['Outcome'])

test_df_norm, normal_mapping = normalize_data(test_df, ignore=['Outcome'])
X_train, Y_train = train_df_norm.drop(['Outcome'], axis=1).values, train_df_norm[['Outcome']].values

X_test, Y_test = test_df_norm.drop(['Outcome'], axis=1).values, test_df_norm[['Outcome']].values

print(X_train.shape)

print(X_test.shape)
class LogisticRegressionModel(torch.nn.Module):

    def __init__(self, feature_count):

        super(LogisticRegressionModel, self).__init__()

        self.linear = torch.nn.Linear(feature_count, 1)

    def forward(self, x):

        return torch.sigmoid(self.linear(x))

    def fit_model(self, X, Y, 

                  learning_rate_a= 0.0001,  learning_rate_b= 0.00001, cut_learning_rate=0.45,

                  epochs=20000, 

                  lambda1 = 0.5, lambda2 = 0.01):

        

        x = torch.autograd.Variable(torch.tensor(X).type(torch.FloatTensor))

        y = torch.autograd.Variable(torch.tensor(Y).type(torch.FloatTensor))

        criterion = torch.nn.BCELoss()

        optimizer = torch.optim.Adamax(

            self.parameters(), lr=learning_rate_a, weight_decay=1e-6)

        losses = []

        using_LR = "A"

        

        for epoch in range(epochs):

            optimizer.zero_grad()

            # ===================forward=====================

            output = self.forward(x)

            loss = criterion(output, y)

            # Some Regularization

            all_linear_params = torch.cat([x.view(-1) for x in self.linear.parameters()])

            l1_regularization = lambda1 * torch.norm(all_linear_params, 1)

            l2_regularization = lambda2 * torch.norm(all_linear_params, 2)

            #loss += l1_regularization + l2_regularization

            # ===================backward====================

            loss.backward()

            optimizer.step()

            # ===================log========================

            if (loss.data.item() <= cut_learning_rate):

                using_LR = 'B'

                for param_group in optimizer.param_groups:

                    param_group['lr'] = learning_rate_b

            else:

                using_LR = 'A'

                for param_group in optimizer.param_groups:

                    param_group['lr'] = learning_rate_a



            print('Using LR_'+using_LR+' epoch [{}/{}], loss:{:.8f}'

                  .format(epoch+1, epochs, loss.data.item()), end='\r')

            losses.append(loss.data.item())

            

        q = [x for x in range(len(losses))]

        plt.plot(q, losses)

        
modelA = LogisticRegressionModel(X_train.shape[1])

modelA.fit_model(X_train, Y_train)

# Create Torch Tensor

x = torch.autograd.Variable(torch.tensor(X_test).type(torch.FloatTensor))

# Get Results

modelA_out = modelA(x).detach().numpy()
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve



precisionA, recallA, thresholdsA = precision_recall_curve(Y_test, modelA_out)

fpr_A, tpr_A, threshold_A = roc_curve(Y_test, modelA_out)



fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(10,10))

fig.suptitle('Metrics', fontsize=16)



ax = axes[0]

ax.plot(recallA, precisionA, 'k--', label='Logistic Regression')



ax.set_title("Precision Recall")

ax.set_xlabel("Recall")

ax.set_ylabel("Precision")

ax.set_ylim([0.0, 1.05])

ax.set_xlim([0.0, 1.0])

ax.set_aspect(1.0)

ax.grid()





ax = axes[1]

ax.plot(fpr_A, tpr_A, 'k--', label='Logistic Regression')

ax.set_title("ROC")

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_ylim([0.0, 1.05])

ax.set_xlim([0.0, 1.0])

ax.set_aspect(1.0)



ax.grid()

plt.show()
th = threshold_A[np.where(fpr_A >= 0.3)[0][0]]

print('Selected Threshold',th)

test_results = test_df[['Outcome']]

test_results.columns = ['Actual']

test_results['Model Porba'] = modelA_out

test_results['Model Prediction'] = test_results['Model Porba'].apply(lambda x: 1 if (x > th) else 0)

test_results.head(10)
pd.crosstab(test_results['Model Prediction'].apply(lambda x: "Yes" if x == 1 else "No"), 

            test_results.Actual.apply(lambda x: "Yes" if x == 1 else "No"))
from sklearn.metrics import accuracy_score

print("Model Accuracy", str("%.2f"%(100*accuracy_score(test_results.Actual, test_results['Model Prediction']))))