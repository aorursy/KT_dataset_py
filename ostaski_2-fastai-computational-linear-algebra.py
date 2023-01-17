import numpy as np

from sklearn.datasets import fetch_20newsgroups

from sklearn import decomposition

from scipy import linalg

import matplotlib.pyplot as plt



%matplotlib inline

np.set_printoptions(suppress=True)
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

remove = ('headers', 'footers', 'quotes')

# fix for SSL handshake error: http://thomas-cokelaer.info/blog/2016/01/python-certificate-verified-failed/

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Captain Obvious reminds us to enable Internet ;)

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)

newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)
newsgroups_train.filenames.shape, newsgroups_train.target.shape
print("\n".join(newsgroups_train.data[:3]))
np.array(newsgroups_train.target_names)[newsgroups_train.target[:3]]
newsgroups_train.target[:10]
num_topics, num_top_words = 6, 8
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = CountVectorizer(stop_words='english')

vectors = vectorizer.fit_transform(newsgroups_train.data).todense() # (documents, vocab)

vectors.shape # vectors.nnz / vectors.shape[0], row_means.shape
print(len(newsgroups_train.data), vectors.shape)
vocab = np.array(vectorizer.get_feature_names())

vocab.shape
vocab[7000:7020]
%time U, s, Vh = linalg.svd(vectors, full_matrices=False) # %time is line magic
print(U.shape, s.shape, Vh.shape)
#Exercise: confirm that U, s, Vh is a decomposition of the var Vectors

reconstructed_vectors = U @ np.diag(s) @ Vh

np.linalg.norm(reconstructed_vectors - vectors)

np.allclose(reconstructed_vectors, vectors)
#Exercise: Confirm that U, Vh are orthonormal

np.allclose(U.T @ U, np.eye(U.shape[0]))

np.allclose(Vh @ Vh.T, np.eye(Vh.shape[0]))
plt.plot(s);
plt.plot(s[:10])
num_top_words=8



def show_topics(a):

    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]

    topic_words = ([top_words(t) for t in a])

    return [' '.join(t) for t in topic_words]
show_topics(Vh[:10])
m,n=vectors.shape

d=5  # num topics
# pegs CPU usage to almost 200%

clf = decomposition.NMF(n_components=d, random_state=1)



W1 = clf.fit_transform(vectors)

H1 = clf.components_
show_topics(H1)
vectorizer_tfidf = TfidfVectorizer(stop_words='english')

vectors_tfidf = vectorizer_tfidf.fit_transform(newsgroups_train.data) # (documents, vocab)
W1 = clf.fit_transform(vectors_tfidf)

H1 = clf.components_
show_topics(H1)
plt.plot(clf.components_[0])
clf.reconstruction_err_
lam=1e3

lr=1e-2

m, n = vectors_tfidf.shape


# pegs CPU usage to almost 200%

W1 = clf.fit_transform(vectors)

H1 = clf.components_
show_topics(H1)
mu = 1e-6

def grads(M, W, H):

    R = W@H-M

    return R@H.T + penalty(W, mu)*lam, W.T@R + penalty(H, mu)*lam # dW, dH
def penalty(M, mu):

    return np.where(M>=mu,0, np.min(M - mu, 0))
def upd(M, W, H, lr):

    dW,dH = grads(M,W,H)

    W -= lr*dW; H -= lr*dH
def report(M,W,H): 

    print(np.linalg.norm(M-W@H), W.min(), H.min(), (W<0).sum(), (H<0).sum())
W = np.abs(np.random.normal(scale=0.01, size=(m,d)))

H = np.abs(np.random.normal(scale=0.01, size=(d,n)))
report(vectors_tfidf, W, H)
upd(vectors_tfidf,W,H,lr)
report(vectors_tfidf, W, H)
# pegs CPU usage to ~ 150%+

for i in range(50): 

    upd(vectors_tfidf,W,H,lr)

    if i % 10 == 0: report(vectors_tfidf,W,H)
show_topics(H)
# old notes (w/o GPU)

# See https://github.com/pytorch/pytorch/issues/1668

#if torch.cuda.available(): # AttributeError: module 'torch.cuda' has no attribute 'available'

#    import torch.cuda as t

#else:

#    import torch as t

import torch

import torch.cuda as tc

from torch.autograd import Variable
def V(M): return Variable(M, requires_grad=True)
v=vectors_tfidf.todense()
#t_vectors = torch.Tensor(v.astype(np.float32))#.cuda()

t_vectors = torch.Tensor(v.astype(np.float32)).cuda()
mu = 1e-5
def grads_t(M, W, H):

    R = W.mm(H)-M

    return (R.mm(H.t()) + penalty_t(W, mu)*lam, 

        W.t().mm(R) + penalty_t(H, mu)*lam) # dW, dH



def penalty_t(M, mu):

    return (M<mu).type(tc.FloatTensor)*torch.clamp(M - mu, max=0.)



def upd_t(M, W, H, lr):

    dW,dH = grads_t(M,W,H)

    W.sub_(lr*dW); H.sub_(lr*dH)



def report_t(M,W,H): 

    print((M-W.mm(H)).norm(2), W.min(), H.min(), (W<0).sum(), (H<0).sum())
# old

# See AssertionError: Torch not compiled with CUDA enabled or import torch vs torch.cuda as t (see imports above)

#t_W = tc.FloatTensor(m,d)

t_W = tc.FloatTensor(m,d)

t_H = tc.FloatTensor(d,n)

t_W.normal_(std=0.01).abs_(); 

t_H.normal_(std=0.01).abs_();



# old

#d=6; lam=100; lr=0.05

# checking torch version for next cell

#print(torch.__version__) # 1.0.0.dev20181205

# downgraded torch with: 'conda install pytorch=0.4.0 cuda92 -c pytorch'

# See https://forums.fast.ai/t/runtime-error-cannot-initialize-aten-cuda-library/20378/5

d=6; lam=100; lr=0.05
# old

#%%time # I let this run for 42 hours before shutting down the kernel



# First time this cell runs we get an error:

#AssertionError: 

#Found no NVIDIA driver on your system. Please check that you

#have an NVIDIA GPU and installed a driver from

#http://www.nvidia.com/Download/index.aspx

for i in range(1000): 

    upd_t(t_vectors,t_W,t_H,lr)

    if i % 100 == 0: 

        report_t(t_vectors,t_W,t_H)

        lr *= 0.9
show_topics(t_H.cpu().numpy())
plt.plot(t_H.cpu().numpy()[0])
t_W.mm(t_H).max()
t_vectors.max()
x = Variable(torch.ones(2, 2), requires_grad=True)

print(x)
print(x.data)



print(x.grad)



y = x + 2

print(y)
z = y * y * 3

out = z.sum()

print(z, out)
out.backward()

print(x.grad)



lam=1e6



# 0ld

# Got that 'AssertionError: Found no NVIDIA driver on your system.' once sometimes twice, then it worked?!?

# This is taking forever as was the for loop between cells 51 and 52 ... lack of GPU issue? Pleanty of RAM and swap available.

# Skipping this after a few weeks

pW = Variable(tc.FloatTensor(m,d), requires_grad=True)

pH = Variable(tc.FloatTensor(d,n), requires_grad=True)

pW.data.normal_(std=0.01).abs_()

pH.data.normal_(std=0.01).abs_();
def report():

    W,H = pW.data, pH.data

    print((M-pW.mm(pH)).norm(2).data[0], W.min(), H.min(), (W<0).sum(), (H<0).sum())



def penalty(A):

    return torch.pow((A<0).type(tc.FloatTensor)*torch.clamp(A, max=0.), 2)



def penalize(): return penalty(pW).mean() + penalty(pH).mean()



def loss(): return (M-pW.mm(pH)).norm(2) + penalize()*lam
M = Variable(t_vectors).cuda()
opt = torch.optim.Adam([pW,pH], lr=1e-3, betas=(0.9,0.9))

lr = 0.05

report()
for i in range(1000): 

    opt.zero_grad()

    l = loss()

    l.backward()

    opt.step()

    if i % 100 == 99: 

        report()

        lr *= 0.9 # learning rate annealling
h = pH.data.cpu().numpy()

show_topics(h)
plt.plot(h[0]);
vectors.shape
%time U, s, Vh = linalg.svd(vectors, full_matrices=False)
print(U.shape, s.shape, Vh.shape)
%time u, s, v = decomposition.randomized_svd(vectors, 5)
%time u, s, v = decomposition.randomized_svd(vectors, 5)
u.shape, s.shape, v.shape
show_topics(v)
from scipy import linalg
# computes an orthonormal matrix whose range approximates the range of A

# power_iteration_normalizer can be safe_sparse_dot (fast but unstable), LU (in between), or QR (slow but most accurate)

def randomized_range_finder(A, size, n_iter=5):

    Q = np.random.normal(size=(A.shape[1], size))

    

    for i in range(n_iter):

        Q, _ = linalg.lu(A @ Q, permute_l=True)

        Q, _ = linalg.lu(A.T @ Q, permute_l=True)

        

    Q, _ = linalg.qr(A @ Q, mode='economic')

    return Q
def randomized_svd(M, n_components, n_oversamples=10, n_iter=4):

    

    n_random = n_components + n_oversamples # see How should we choose r? cell below for n_oversamples rationale

    

    Q = randomized_range_finder(M, n_random, n_iter)

    

    # project M to the (k + p) dimensional space using the basis vectors

    B = Q.T @ M

    

    # compute the SVD on the thin matrix: (k + p) wide

    Uhat, s, V = linalg.svd(B, full_matrices=False)

    del B

    U = Q @ Uhat

    

    return U[:, :n_components], s[:n_components], V[:n_components, :]


# pegs CPU usage to ~150%

u, s, v = randomized_svd(vectors, 5)
%time u, s, v = randomized_svd(vectors, 5)
u.shape, s.shape, v.shape
show_topics(v)
#Exercise: Write a loop to calculate the error of your decomposition as you vary the # of topics

# pegs CPU usage between 100% to 200%

step = 20

n = 20

error = np.zeros(n)



for i in range(n):

    U, s, V = randomized_svd(vectors, i * step)

    reconstructed = U @ np.diag(s) @ V

    error[i] = np.linalg.norm(vectors - reconstructed)
plt.plot(range(0,n*step,step),error)
%time u, s, v = decomposition.randomized_svd(vectors, 5)
%time u, s, v = decomposition.randomized_svd(vectors.todense(), 5)
type(vectors)