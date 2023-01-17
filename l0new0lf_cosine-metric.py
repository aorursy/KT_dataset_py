import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(8,5))



plt.arrow(0,0,2,2, head_width=0.3, length_includes_head=True, color='blue', alpha=1)

plt.arrow(0,0,3,0, head_width=0.3, length_includes_head=True, color='red')

plt.arrow(0,0,0,3, head_width=0.3, length_includes_head=True, color='blue', alpha=1)

plt.arrow(0,0,-2,-2, head_width=0.3, length_includes_head=True, color='blue', alpha=1)

plt.arrow(0,0,2,-2, head_width=0.3, length_includes_head=True, color='blue', alpha=1)

plt.arrow(0,0,-3,0, head_width=0.3, length_includes_head=True, color='blue', alpha=1)



plt.text(3.1, 0, 'Reference vector')

plt.text(2, 2, 'Similarity = cos(45) = +0.7')

plt.text(0, 3, 'Similarity = cos(90) = 0')

plt.text(-5, -2.5, 'Similarity = cos(225) = -0.7')

plt.text(2, -2, 'Similarity = cos(315) = +0.7')

plt.text(-6, 0.5, 'Similarity = cos(180) = -1 \n(completely opposite)')



plt.plot(6,4)

plt.plot(0,0)

plt.plot(-6,-4)

plt.grid()

plt.title("Similarity of vectors\n Range: [-1, 1]")



plt.show()
plt.figure(figsize=(8,5))



plt.arrow(0,0,2,2, head_width=0.3, length_includes_head=True, color='blue', alpha=1)

plt.arrow(0,0,5,5, head_width=0.3, length_includes_head=True, color='yellow', alpha=0.5)

plt.arrow(0,0,1,3, head_width=0.3, length_includes_head=True, color='red')



plt.text(2,1.5,'X_1')

plt.text(5,4.5,'X_2')

plt.text(1,2.5,'X_3')



plt.plot(6,6)

plt.plot(0,0)

plt.grid()

plt.title("Each data sample is a Position Vector")



plt.show()
class Distance:

    @staticmethod

    def _norm(X, p):

        # raise_by_p 

        vec2scalar = np.sum(np.power(X, p)) 

        # squash_by_p 

        norm = np.power(vec2scalar, (1/p))

        return norm

    

    @staticmethod

    def __cosine_sim(X1, X2, norm_p=2):

        X1_dot_X2     = X1.T.dot(X2)

        norm_X1  = Distance._norm(X1, norm_p)

        norm_X2  = Distance._norm(X2, norm_p)

        return (X1_dot_X2) / (norm_X1*norm_X2)



    @staticmethod

    def cosine(X1, X2):

        return 1 - Distance.__cosine_sim(X1, X2)

    

    @staticmethod

    def euclidean(X1, X2):

        diff_vec = X2 - X1

        return Distance._norm(diff_vec, 2)
import numpy as np



# Note: vectors are just points (dir, vecs from origin)

# =====================================================



# test perpendicular vecs (not unity)

a = np.array([0,3])

b = np.array([3,0])

print('perpclr vecs: \t', Distance.cosine(a, b))



# test parallel vecs (not unity)

a = np.array([0,3])

b = np.array([0,9])

print('parallel vecs: \t', Distance.cosine(a, b))



# test opposite vecs (not unity)

a = np.array([0,3])

b = np.array([0,-9])

print('opposite vecs: \t', Distance.cosine(a, b))



# test random vecs (not unity)

a = np.array([0,3])

b = np.array([5,4])

print('random vecs: \t', Distance.cosine(a, b))
import numpy as np



print("""Note: Euclidean dist. doesnt capture direction

It captures norm(size) of individual diffs of components\n""") 

# =====================================================



# test perpendicular vecs (not unity)

a = np.array([0,3])

b = np.array([3,0])

print('perpclr vecs: \t', Distance.euclidean(a, b))



# test parallel vecs (not unity)

a = np.array([0,3])

b = np.array([0,99])

print('parallel vecs: \t', Distance.euclidean(a, b))



# test opposite vecs (not unity)

a = np.array([0,3])

b = np.array([0,-9])

print('opposite vecs: \t', Distance.euclidean(a, b))



# test random vecs (not unity)

a = np.array([0,3])

b = np.array([5,4])

print('random vecs: \t', Distance.euclidean(a, b))
def get_vec_w_dir_cosines(vector_w_dir_ratios): 

    return vector_w_dir_ratios / Distance._norm(vector_w_dir_ratios, 2)
a = np.array([2,3,10])

b = np.array([5,4,-8])



# satisfy conditions (to dir cosines)

a_w_dir_cosines = get_vec_w_dir_cosines(a)

b_w_dir_cosines = get_vec_w_dir_cosines(b)



print('Unit vectors:')

print('===================================================')

print(a_w_dir_cosines)

print(b_w_dir_cosines)

print('===================================================')



cos_dist = Distance.cosine(a_w_dir_cosines, b_w_dir_cosines)

euc_dist = Distance.euclidean(a_w_dir_cosines, b_w_dir_cosines)



print('cos_dist: \t', cos_dist, '\t 2 x cos_dist \t', 2*cos_dist)

print('euc_dist: \t', euc_dist, '\t euc_dist ^2 \t', euc_dist**2)