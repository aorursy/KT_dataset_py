#!pip install nb_black

#%load_ext nb_black

import numpy as np

np.random.seed(42)
# Distribution

distribution = {

    "mean": np.random.rand(1, 3),

    "covariance": [[1, 0.1, 0], [0.1, 1, 0], [0, 0, 1]],

}



# Point

point = np.random.rand(1, 3)





def mahalanobis_distance(distribution: "dict", point: "np.array()") -> int:

    """ Estimate Mahalanobis Distance 

    

    Args:

        distribution: a sample gaussian distribution

        point: a deterministic point

    

    Returns:

        Mahalanobis distance

    """

    mean = distribution["mean"]

    cov = distribution["covariance"]

    return np.sqrt((point - mean) @ np.linalg.inv(cov) @ (point - mean).T)[0][0]





# Our implementation

distance = mahalanobis_distance(distribution, point)

print(f"Ours : {distance}")



# scipy inbuilt

from scipy.spatial.distance import mahalanobis



distance = mahalanobis(

    point, distribution["mean"], np.linalg.inv(distribution["covariance"])

)

print(f"Scipy: {distance}")
# Distribution 1

distribution1 = {

    "mean": np.array([[1,3,1]]),

    "covariance": np.array([[1, 0.1, 0], [0.1, 1, 0], [0, 0, 1]]),

}





# Distribution 2

distribution2 = {

    "mean": np.array([[1,3,1]]),

    "covariance": np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]),

}



# Distribution 3

d1 = np.random.rand(1,1000)

p1 = np.histogram(d1,100)[0]

p1 = p1 / np.sum(p1)



# Distribution 4

d2 = np.random.rand(1,1000)

p2 = np.histogram(d2,100)[0]

p2 = p1 / np.sum(p2)



def bhattacharyya_gaussian_distance(distribution1: "dict", distribution2: "dict",) -> int:

    """ Estimate Bhattacharyya Distance (between Gaussian Distributions)

    

    Args:

        distribution1: a sample gaussian distribution 1

        distribution2: a sample gaussian distribution 2

    

    Returns:

        Bhattacharyya distance

    """

    mean1 = distribution1["mean"]

    cov1 = distribution1["covariance"]



    mean2 = distribution2["mean"]

    cov2 = distribution2["covariance"]



    cov = (1 / 2) * (cov1 + cov2)



    T1 = (1 / 8) * (

        np.sqrt((mean1 - mean2) @ np.linalg.inv(cov) @ (mean1 - mean2).T)[0][0]

    )

    T2 = (1 / 2) * np.log(

        np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))

    )



    return T1 + T2



def bhattacharyya_distance(distribution1: "dict", distribution2: "dict",) -> int:

    """ Estimate Bhattacharyya Distance (between General Distributions)

    

    Args:

        distribution1: a sample distribution 1

        distribution2: a sample distribution 2

    

    Returns:

        Bhattacharyya distance

    """

    sq = 0

    for i in range(len(distribution1)):

        sq  += np.sqrt(distribution1[i]*distribution2[i])

    

    return -np.log(sq)

    

    

# Our implementation (Gaussian)

distance = bhattacharyya_gaussian_distance(distribution1, distribution2)

print(f"Ours (Gaussian) : {distance}")



# Our implementation (General)

distance = bhattacharyya_distance(p1, p2)

print(f"Ours (General)  : {distance}")

# Distribution 1

d1 = np.random.rand(1,1000)

p1 = np.histogram(d1,100)[0] + 0.000001

p1 = p1 / np.sum(p1)



# Distribution 2

d2 = np.random.randn(1,1000)

p2 = np.histogram(d2,100)[0] + 0.000001

p2 = p2 / np.sum(p2)





def KL_divergence(distribution1: "dict", distribution2: "dict",) -> int:

    """ Estimate KL-Divergence (from distribution1 to distribution2)

    

    Args:

        distribution1: a sample distribution 1

        distribution2: a sample distribution 2

    

    Returns:

        KL-Divergence distance

    """

    s = 0

    for i in range(len(distribution2)):

        p = distribution2[i]

        q = distribution1[i]

        s += p*np.log(q/p)

    

    return s

        

# Our implementation

distance = KL_divergence(p1, p2)

print(f"Ours (1 (Uniform) ->2 (Gaussian)): {distance}")

    

distance = KL_divergence(p2, p1)

print(f"Ours (2 (Gaussian)->1 (Uniform)): {distance}")

    
