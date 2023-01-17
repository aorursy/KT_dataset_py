import numpy as np 
x1=np.vstack(np.array([1,-1,1,1])).T

y1=np.vstack(np.array([1,-1,1,-1])).T

W1=np.matmul(x1.T,y1)

x1
y1
x2=np.vstack(np.array([-1,1,1,-1])).T

y2=np.vstack(np.array([-1,-1,1,-1])).T

W2=np.matmul(x2.T,y2)
x2
y2
W=W1+W2

W
W__=lambda p=(x1,y1) : np.matmul(p[0].T,p[1])

W_=lambda list_pair=[(x1,y1),(x2,y2)] : 0 if len(list_pair)==0 else W__(list_pair.pop())+W_(list_pair)
W=W_([(x1,y1),(x2,y2)])

W
sgn=lambda x : (x>0)*1 - (x<0)*1
sgn(W)
E=lambda x,y,W=W : -1/2*np.matmul(np.matmul(x,W),y.T).item()
E(x1,y1)
y_=lambda x,W=W : sgn(np.matmul(x,W))
y_(x1)==y1
x0=np.vstack(np.array([-1,-1,1,-1])).T
E(x0,y_(x0))
x1=np.vstack(np.array([-1,-1,1,1])).T

x2=np.vstack(np.array([1,-1,1,-1])).T
W__h=lambda x : np.matmul(x.T,x)

W_h=lambda list_pairs : 0 if len(list_pairs)==0 else W__h(list_pairs.pop())+W_h(list_pairs)
def W_hope(list_pairs):

    W=W_h(list_pairs)

    for i in range(W.shape[0]):

        W[i,i]=0

    return(W)

W=W_hope([x1,x2])

W
x=np.vstack(np.array([-1,-1,1,1])).T
class Oja : 

    def __init__(self):

        self.w=np.vstack(np.array([1,1],dtype='float64'))

    def input_feature_space(self,list_):

        self.input_feature_space=[]

        for l in list_:

            self.input_feature_space.append(np.vstack(np.array(l,dtype='float64')))

        self.barycenter=self._barycenter()

    def _barycenter(self):

        return(sum(self.input_feature_space)/len(self.input_feature_space))

    def update_weight(self,x,lam):

        x-=self.barycenter # Consider a center feature space ... we look for w in a trigonometric circle

        phi=np.matmul(x.T,self.w).item()

        self.w=self.w+lam*phi*(x-phi*self.w)

a_oja=Oja()
a_oja.w
a_oja.input_feature_space([[4,5],[5,4],[2,1],[1,2]])
a_oja.barycenter.T
a_oja.update_weight(a_oja.input_feature_space[0],0.1)
a_oja.w
dist=lambda a,b : np.linalg.norm(a-b) # Grid metric 

neighborhood = lambda k,r=1,bound=[0,5] : [k-i for i in range(1,r+1) if k-i >=0]+[k]+[k+i for i in range (1,r+1) if k+i<=5]

phi=lambda k,i,r=1,bound=[0,5] : 1 if i in neighborhood(k,r,bound) else 0 # Lateral interaction function 
class Kmap:

    def __init__(self,l_w,r=0):

        self.w=np.array(l_w,dtype='float64')

        self.r=r

    def input_feature_space(self,list_):

        self.input_feature_space=[]

        for l in list_:

            self.input_feature_space.append(np.vstack(np.array(l)))

    def _winner_talk(self,x,dist=dist):

        ar_dist=np.apply_along_axis(lambda w : dist(w,x.T),arr=self.w,axis=1)

        print('dist : ',ar_dist)

        winner=ar_dist.argmin()

        min_dist=ar_dist[winner]

        return(winner,min_dist)

    def update_weight(self,x,alpha,dist=dist,phi=phi):

        phi_=lambda k,i : phi(k,i,r=self.r,bound=[0,self.w.shape[0]-1])

        winner,min_dist=self._winner_talk(x,dist)

        for i,w in enumerate(self.w): # Itterate over the row 

            print('i :',i,'(x.T-w[:])',(x.T-w[:])[0])

            self.w[i]+=alpha*phi_(winner,i)*(x.T-w[:])[0]

        return(self.w)

n_kmap=Kmap([[3,-1,2],[-3,5,1]])
print(n_kmap.w)
x=np.vstack(np.array([2,1,-3]))
n_kmap._winner_talk(x)
n_kmap.update_weight(x,alpha=0.3)
np.array([ 3.,-1.,2.])-0.3*0.15*np.array([-1.,2.,-5.])
class LVQ1:

    def __init__(self,l_w,r=0):

        """l_w : pour la list des weightes 

            l_class : la list de la classe auquelle est associé chaque neurones

            r : radius of the neigthborhood for lvq it's always set to 0 """

        self.w=np.array(l_w,dtype='float64')

        self.r=r

    def input_feature_space(self,list_):

        self.input_feature_space=[]

        for l in list_:

            self.input_feature_space.append(np.vstack(np.array(l)))

    def _winner_talk(self,x,dist=dist):

        ar_dist=np.apply_along_axis(lambda w : dist(w,x.T),arr=self.w,axis=1)

        print('dist : ',ar_dist)

        winners=list(ar_dist.argsort()[0:2])

        return(winners)

    def update_weight(self,x,class_,alpha,dist=dist,phi=phi):

        phi_=lambda k,i : phi(k,i,r=self.r,bound=[0,self.w.shape[0]-1])

        winner1,winner2=self._winner_talk(x,dist)

        for i,w in enumerate(self.w): # Itterate over the row 

            print('i :',i,'(x.T-w[:])',(x.T-w[:])[0])

            if winner1==class_:

                 # Si il est dans la même classe : Rapprocher le neurone le plus proche de l'input 

                self.w[i]+=alpha*phi_(winner1,i)*(x.T-w[:])[0]

            else :

                # Sinon : Eloigner le neurone le plus proche de l'input

                self.w[i]-=alpha*phi_(winner1,i)*(x.T-w[:])[0]

        return(self.w)

lvq=LVQ1([[3,-1,2],[-3,5,1]])
x=np.vstack(np.array([2,1,-3]))
lvq._winner_talk(x)
lvq.update_weight(x,2,0.3)
window=lambda d0,d1,w : ((min(d0/d1,d1/d0)>(1-w)) and (max(d0/d1,d1/d0)>(1+w)))*1

class LVQ2_1:

    def __init__(self,l_w,l_class_=None,r=0):

        """l_w : pour la list des weightes 

            l_class : la list de la classe auquelle est associé chaque neurones

            r : radius of the neigthborhood for lvq it's always set to 0 """

        self.w=np.array(l_w,dtype='float64')

        self.r=r

        self.class_ =list(range(self.w.shape[0])) if l_class_ is None else l_class_

        

    def input_feature_space(self,list_):

        self.input_feature_space=[]

        for l in list_:

            self.input_feature_space.append(np.vstack(np.array(l)))

    def _winner_talk(self,x,dist=dist):

        ar_dist=np.apply_along_axis(lambda w : dist(w,x.T),arr=self.w,axis=1)

        print('dist : ',ar_dist)

        winners=list(ar_dist.argsort()[0:2])

        d0,d1=ar_dist[winners[0]],ar_dist[winners[1]]

        return(winners,[d0,d1])

    

    def update_weight(self,x,class_,alpha,w_,dist=dist,phi=phi):

        """ x: input patern 

            class_ : class of the input patern

            alpha : learning parameter 

            w_ : window parameter 

            dist : distance metric (euclidian by default)

            phi : interconnection function (no interconnection for lvq)"""

        phi_=lambda k,i : phi(k,i,r=self.r,bound=[0,self.w.shape[0]-1])

        winners,dists=self._winner_talk(x,dist)

        winner1,winner2=winners

        print(winners)

        for i,w in enumerate(self.w): # Itterate over the row 

            # Si 

            #       les deux neurones les plus proches ne sont pas dans la même classe,

            #    ET que x est dans la window

            if self.class_[winner1]==class_ and (not slef.class_[winner2]==class_) and window(*dists,w_)==1:

                self.w[i]+=alpha*phi_(winner1,i)*(x.T-w[:])[0] # Rapprocher winner 1

                self.w[i]-=alpha*phi_(winner2,i)*(x.T-w[:])[0] # Eloigner winner 2

            elif self.class_[winner2]==class_ and (not slef.class_[winner1]==class_) and window(*dists,w_)==1:

                self.w[i]+=alpha*phi_(winner2,i)*(x.T-w[:])[0] # Rapprocher winner 2

                self.w[i]-=alpha*phi_(winner1,i)*(x.T-w[:])[0] # Eloigner winner 1

            # Sinon : ne rien faire

            else :

                pass

        return(self.w)
window=lambda d0,d1,w : (min(d0/d1,d1/d0)>(1-w)/(1+w))*1

class LVQ3:

    def __init__(self,l_w,l_class_=None,r=0):

        """l_w : pour la list des weightes 

            l_class : la list de la classes auquel est associé chaque neurones

            r : radius of the neigthborhood for lvq it's always set to 0 """

        self.w=np.array(l_w,dtype='float64')

        self.r=r

        self.class_ =list(range(self.w.shape[0])) if l_class_ is None else l_class_

        

    def input_feature_space(self,list_):

        self.input_feature_space=[]

        for l in list_:

            self.input_feature_space.append(np.vstack(np.array(l)))

    def _winner_talk(self,x,dist=dist):

        ar_dist=np.apply_along_axis(lambda w : dist(w,x.T),arr=self.w,axis=1)

        print('dist : ',ar_dist)

        winners=list(ar_dist.argsort()[0:2])

        d0,d1=ar_dist[winners[0]],ar_dist[winners[1]]

        return(winners,[d0,d1])

    

    def update_weight(self,x,class_,alpha,eps,w_,dist=dist,phi=phi):

        """ x: input patern 

            class_ : class of the input patern

            alpha : learning parameter 

            w_ : window parameter 

            eps : proportion of move for correct vectors

            dist : distance metric (euclidian by default)

            phi : interconnection function (no interconnection for lvq)"""

        phi_=lambda k,i : phi(k,i,r=self.r,bound=[0,self.w.shape[0]-1])

        winners,dists=self._winner_talk(x,dist)

        winner1,winner2=winners

        print(winners)

        for i,w in enumerate(self.w): # Itterate over the row 

            # Si 

            #       les deux neurones les plus proches ne sont pas dans la même classe,

            #    ET que x est dans la window

            if self.class_[winner1]==class_ and (not slef.class_[winner2]==class_) and window(*dists,w_)==1:

                self.w[i]+=alpha*phi_(winner1,i)*(x.T-w[:])[0] # Rapprocher winner 1

                self.w[i]-=alpha*phi_(winner2,i)*(x.T-w[:])[0] # Eloigner winner 2

            elif self.class_[winner2]==class_ and (not slef.class_[winner1]==class_) and window(*dists,w_)==1:

                self.w[i]+=alpha*phi_(winner2,i)*(x.T-w[:])[0] # Rapprocher winner 2

                self.w[i]-=alpha*phi_(winner1,i)*(x.T-w[:])[0] # Eloigner winner 1

            

            # Ou quand les deux plus proches neurones sont dans la même classe

            elif self.class_[winner1]==self.class_[winner2]:

                self.w[i]+=eps*alpha*phi_(winner1,i)*(x.T-w[:])[0]

                self.w[i]+=eps*alpha*phi_(winner2,i)*(x.T-w[:])[0]

            # Sinon ne rien faire

            else :

                pass

        return(self.w)



lvq=LVQ3([[3,-1,2],[-3,5,1]])
x=np.vstack(np.array([2,1,-3]))

winners,dists=lvq._winner_talk(x)
window(*dists,0.15)
lvq.update_weight(x,1,alpha=0.3,eps=0.2,w_=0.15)