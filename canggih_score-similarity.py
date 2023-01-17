import numpy as np
import pandas as pd
def HistogramSimilarity(ar1,ar2):
    # Using chi squared distance
    ar1 = np.array(ar1)
    ar2 = np.array(ar2)
    dis = np.sum((ar1-ar2)**2/(ar1+ar2))
    return dis

def ConstructMatrixScore(data):
    index = data.iloc[:,0]
    score = data.iloc[:,1::]
    dfscore = pd.DataFrame(0.0,index=index, columns=index)
    for i in range(0,len(index)):
        idata = score.iloc[i,1::]
        for j in range(i,len(index)):
            jdata = score.iloc[j,1::]
            if i == j:
                dfscore.iat[i,j] = -1
            else:
                strval = '{:.5f}'.format(HistogramSimilarity(idata,jdata))
                dfscore.iat[i,j] = strval
    mscore = np.matrix(dfscore)
    newscore = mscore + np.transpose(mscore)
    dfnewscore = pd.DataFrame(newscore,index=index, columns=index)
    return dfnewscore
dscore = pd.read_csv('../input/datascorehist-all-share-new.csv',sep='|')
score_matrix = ConstructMatrixScore(dscore)
print(score_matrix.head(10))
score_matrix.to_csv('sim_score.csv')