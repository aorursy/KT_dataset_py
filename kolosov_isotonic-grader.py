import numpy as np
def Dist2Cov(dist):
    """
    Transforms Euclidean distance matrix to covariance matrix of centered configuration
    No diagnostics if input matrix is not Euclidean distance matrix
    dist - (pairwise) Euclidean distance matrix of shape (numpoint,numpoint)
    """
    assert(np.all(dist.diagonal()==0))
    assert(np.all(dist>=0))
    assert(np.all(dist==dist.T))
    n=dist.shape[0]
    D2=dist*dist
    tr=np.sum(D2)/n/2
    diag=(np.sum(D2,axis=0)-tr)/n
    cov=(diag[np.newaxis,:]+diag[:,np.newaxis]-D2)/2
    #assert(np.sum(cov)==0) will fail due to computational errors
    cov-=np.sum(cov)/n/n # some error correction
    assert(np.all(cov.diagonal()>=0))
    assert(np.all(cov==cov.T))
    return cov
def cMDS(dist, epsilon = 1e-9):
    """
    cMDS performs Classical Multidimensional Scaling
    dist - (pairwise) Euclidean distance matrix of shape (numpoint,numpoint)

    Returns
    -------
    isEuclidean - test result
    mindim - minimal dimensionality required for exact embedding
    configuration - centered point configuration of shape (numpoint,mindim)
    eigenValues - sorted, shape (numpoint), sum(eigenValues[mindim:end])/sum(eigenValues) <= epsilon
    eigenVectors - of unit 2-norm
    """
    K = Dist2Cov(dist)
    eigenValues,eigenVectors = np.linalg.eigh(K)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    assert(np.allclose(K,np.dot(eigenVectors,np.dot(np.diag(eigenValues),eigenVectors.T))))
    idx = np.cumsum(eigenValues[::-1])[::-1]/np.sum(eigenValues) > epsilon
    mindim = np.argmin(idx)
    isEuclidean = np.allclose(K,np.dot(eigenVectors[:,:mindim],np.dot(np.diag(eigenValues[:mindim]),eigenVectors[:,:mindim].T)))
    configuration = np.dot(eigenVectors[:,:mindim], np.diag(np.sqrt(eigenValues[:mindim])))
    return isEuclidean, mindim, configuration, eigenValues, eigenVectors
import pandas as pd
# читаем присланный файл
submission_df=pd.read_csv('submission.csv', sep=',')
# проверяем соответствие формату
assert(submission_df.shape==(4950, 2))
assert(np.all(submission_df.columns==['pair', 'distance']))
# считаем, что в файле заявки ключи пар правильные: однократно все от 1_0 до 99_98 - Kaggle должен был это проверить
# но на всякий случай восстановим их порядок
# если Kaggle сам восстанавливает порядок по ключам, то это не требуется
pairs=submission_df['pair'].str.split('_').to_list()
pairs=[list(map(int,p)) for p in pairs]
stri1=[p[0] for p in pairs]
stri2=[p[1] for p in pairs]
# восстанавливаем матрицу попарных расстояний с учетом порядка строк в исходном файле
submission_dist=np.zeros((100,100))
submission_dist[stri1,stri2]=submission_df['distance']
submission_dist[stri2,stri1]=submission_df['distance']
# вытянем квадрат в вектор
# submission_df['distance'] тоже был вектором, но теперь гарантируем нужный нам порядок
tri1,tri2=np.tril_indices_from(submission_dist,k=-1)
submission_dist_vec=submission_dist[tri1,tri2]
# читаем исходные расстояния, выданные нами на конкурс
actual_df=pd.read_csv('submissionYv1.csv', sep=',')
# проверяем соответствие формату
assert(actual_df.shape==(4950, 2))
assert(np.all(actual_df.columns==['pair', 'distance']))
# мы уверены, что в нашем файле порядок ключей верный, поэтому не сортируем
actual_dist_vec=actual_df['distance'].to_numpy()
assert(np.unique(actual_dist_vec).shape[0]==4950)
from scipy.stats import rankdata
from scipy.stats import pearsonr,spearmanr
# проверяем совпадение рангов
assert(np.unique(submission_dist_vec).shape[0]==4950)
assert(np.all(rankdata(actual_dist_vec)==rankdata(submission_dist_vec)))
pearsonr(rankdata(actual_dist_vec), rankdata(submission_dist_vec))
spearmanr(actual_dist_vec,submission_dist_vec)
isEuclidean, mindim, _, _, _ = cMDS(submission_dist)
isEuclidean, mindim