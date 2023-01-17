# General
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (16, 8)
#import scikitplot as skplt
from sklearn import metrics

save = True
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Hard / Soft for Train / Test
train_hard_pred_matrix = pd.read_csv("../input/titanic-voting-pipeline-stack-and-guide/train_hard_pred_matrix.csv", index_col='PassengerId')
train_soft_pred_matrix = pd.read_csv("../input/titanic-voting-pipeline-stack-and-guide/train_soft_pred_matrix.csv", index_col='PassengerId')
test_hard_pred_matrix = pd.read_csv("../input/titanic-voting-pipeline-stack-and-guide/test_hard_pred_matrix.csv", index_col='PassengerId')
test_soft_pred_matrix = pd.read_csv("../input/titanic-voting-pipeline-stack-and-guide/test_soft_pred_matrix.csv", index_col='PassengerId')

# CV and Validation from my model building
results = pd.read_csv("../input/titanic-voting-pipeline-stack-and-guide/titanic_clf_results.csv")

# Subset top models, excluding previous voting ensembles
results = results.loc[(~results["Model"].str.contains("Voting"))
                      &(results["Model"] != "XGBsklearn")
                      &(results["Model"] != "stacked")
                      &(results["CV Mean"] >= 0.8048),:]
# View Results
results
def topmodels(df, isin=results["Model"]):
    return [x for x in df if x in list(isin)+["Survived"]]

# Subset Models
train_hard_pred_matrix = train_hard_pred_matrix.loc[:,topmodels(df=train_hard_pred_matrix)]
test_hard_pred_matrix = test_hard_pred_matrix.loc[:,topmodels(df=test_hard_pred_matrix)]

train_hard_pred_matrix.sample(5)
np.corrcoef(train_hard_pred_matrix["Random_Forest"], train_hard_pred_matrix["Survived"])[0, 1]
print("Mathew Corr: ",metrics.matthews_corrcoef(train_hard_pred_matrix["Random_Forest"],
                          train_hard_pred_matrix["Survived"]))
print("Pandas Corr: ",train_hard_pred_matrix["Random_Forest"].corr(train_hard_pred_matrix["Survived"]))
def accuracy_matrix(data):
    df = pd.DataFrame(columns = data.columns, index = data.columns)
    for row in data.columns:
        for col in data.columns:
            df.loc[row,col] = metrics.accuracy_score(data.loc[:,row],data.loc[:,col]).astype(float)
    for x in df.columns:
        df[x] = df[x].astype(float)
    return df

def acc_n_corr(df):
    # Train Hard Correlation, All Models
    f,ax = plt.subplots(1,2, figsize=[18,5])
    sns.heatmap(df.corr(),annot=True,cmap="coolwarm",cbar_kws={'label': 'Correlation Coefficient'},ax=ax[0])
    ax[0].set_title("Correlation Matrix")

    # Hard Train Accuracy Matrix, All Models
    sns.heatmap(accuracy_matrix(df),annot=True,cmap="coolwarm",cbar_kws={'label': 'Accuracy Score'}
               ,ax=ax[1])
    ax[1].set_title("Accuracy Score Matrix")
acc_n_corr(df=train_hard_pred_matrix)
plt.show()
acc_n_corr(df=test_hard_pred_matrix)
plt.show()
def voter(df, test_df,results=results.loc[:,"Model"],top=1, nsecond=2, target="Survived", submit= False):
    ref = accuracy_matrix(df.drop(target,axis=1))
    base = [list(results)[top]]
    additional = list(ref[base].sort_values(by=base)[:nsecond].index)
    vote = df[base+additional].mode(axis=1).iloc[:,0]
    score = metrics.accuracy_score(vote, df[target])
    test_vote = pd.DataFrame(test_df[base+additional].mode(axis=1).iloc[:,0]).rename(columns={0:"Survived"})
    
    if save == True & submit == True:
        test_vote.to_csv("{}_{}.csv".format(base[0],nsecond))
    
    return base, additional, test_vote, score
# Save
output = pd.DataFrame(columns=["Baseline","Secondary_Models","Score", "Base_rank","Add_num"])

# Iterate
for x in range(len(list(results["Model"]))):
    for y in range(1,len(list(results["Model"]))-1):
        base, additional, temp, score = voter(df=train_hard_pred_matrix,test_df=test_hard_pred_matrix, results=results["Model"],
                    top=x, nsecond=y, target="Survived", submit=False)
        output = output.append({'Baseline': base,"Secondary_Models": additional,"Score": score,
                               "Base_rank": x,"Add_num": y}, ignore_index=True)
# Submit
for x,y in output.sort_values(by="Score", ascending=False).loc[:,["Base_rank","Add_num"]].values[:5]:
    voter(df=train_hard_pred_matrix,test_df=test_hard_pred_matrix, results=results["Model"],top=x,
                    nsecond=y, target="Survived", submit=True)
# View
output.sort_values(by="Score", ascending=False)[:5]
def roc_AUC(data):
    df = pd.DataFrame(columns = data.columns, index = data.columns)
    for row in data.columns:
        for col in data.columns:
            df.loc[row,col] = metrics.roc_auc_score(data.loc[:,row],data.loc[:,col]).astype(float)
    for x in df.columns:
        df[x] = df[x].astype(float)
    return df

def prob_n_corr(df):
    # Train Hard Correlation, All Models
    f,ax = plt.subplots(1,2, figsize=[18,5])
    sns.heatmap(df.corr(),annot=True,cmap="coolwarm",cbar_kws={'label': 'Correlation Coefficient'},ax=ax[0])
    ax[0].set_title("Correlation Matrix")

    # Hard Train Accuracy Matrix, All Models
    sns.heatmap(roc_AUC(df),annot=True,cmap="coolwarm",cbar_kws={'label': 'ROC_AUC Score'}
               ,ax=ax[1])
    ax[1].set_title("ROC AUC Score Matrix")
acc_n_corr(df=train_hard_pred_matrix)
plt.show()
acc_n_corr(df=test_hard_pred_matrix)
plt.show()
# set up cutoff threshold for lower and upper bounds, easy to twist 
cutoff_lo = 0.8
cutoff_hi = 0.2

# Gather Descriptive Statistics for Stacking
def manual_stack(data):
    df = pd.DataFrame()
    df['max'] = data.max(axis=1) # axis = By Row
    df['min'] = data.min(axis=1)
    df['mean'] = data.mean(axis=1)
    df['median'] = data.median(axis=1)
    #df.index = data.index
    return df
test_soft_stack = manual_stack(data=test_soft_pred_matrix)
train_soft_pred_matrix = train_soft_pred_matrix.loc[:,topmodels(df=train_soft_pred_matrix)]
test_soft_pred_matrix = test_soft_pred_matrix.loc[:,topmodels(df=test_soft_pred_matrix)]
def simple_submit(df):
    stacks = manual_stack(df)
    for x in stacks.columns:
        stacks[[x]].rename(columns = {x:"Survived"}, inplace=True)
        if save == True:
            stacks[[x]].to_csv("{}_simple_stack.csv".format(x))
simple_submit(df=test_soft_pred_matrix)
test_soft_pred_matrix.head()
def pushout_median(df, median):
    temp = pd.DataFrame(np.where(np.all(df > cutoff_lo, axis=1), 1, 
                    np.where(np.all(df < cutoff_hi, axis=1),
                    0, median)), columns=["Survived"]).set_index(df.index)
    temp.iloc[:,0] = round(temp.iloc[:,0]).astype(int)
    return temp
prob_eval = pd.DataFrame()
pushoutmedian = pushout_median(df=test_soft_pred_matrix, median=test_soft_stack["median"])
prob_eval["pushout_median"] = pushoutmedian.iloc[:,0]
if save == True:
    pushoutmedian.to_csv("pushout_median.csv")
def stack_evaluation(data, eval_set):
    stack_results = pd.DataFrame()
    modelname = data.columns
    acc = metrics.accuracy_score(data, eval_set)
    stack_results = results.append({'Model': "Stack_{}".format(modelname),
                                    'Test_Score': acc,}, ignore_index=True)
    return stack_results