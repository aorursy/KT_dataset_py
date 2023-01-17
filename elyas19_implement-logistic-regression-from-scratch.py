import numpy as np

import matplotlib.pyplot as plt
mu1 = [7,7]

sigma1 = [[8,3],[3,2]]

c1 = np.random.multivariate_normal(mu1,sigma1,300)



mu2 = [15,6]

sigma2 = [[2,0],[0,2]]

c2 = np.random.multivariate_normal(mu2,sigma2,100)



plt.scatter(c1[:,0],c1[:,1],s= 9,color='blue',marker='*',label='class1')

plt.scatter(c2[:,0],c2[:,1],s= 9,color='orange',marker='o',label='class2')



#################################################################################################

# Batch Gradient Descent

phi = np.concatenate((c1,c2),axis=0)

x = np.concatenate((np.ones((np.size(phi,0),1)),phi),axis=1)

y = np.concatenate((np.ones((np.size(c1,0),1)),np.zeros((np.size(c2,0),1))))



max_ite = 5000; weight_vec = np.random.randn(3,1); lr = 0.1

cost_val_batch = np.zeros((max_ite,1))



for i in range(max_ite):

    # logistic function in vectorized form

    h_x = 1/(1+np.exp(-x.dot(weight_vec)))

    #log-loss a.k.a cost function

    cost_val_batch[i] = -(1/(np.size(x,0)))*sum(y*np.log(h_x) + (1-y)*np.log(1-h_x))

    #gradient of cost function

    grad = (1/(np.size(x,0)))*(x.T.dot((h_x-y)))

    #update weight vectors

    weight_vec = weight_vec - lr*grad



x_plot = np.linspace(5,20,7)

y_plot = -weight_vec[0]/weight_vec[2] - (weight_vec[1]/weight_vec[2])*x_plot

plt.plot(x_plot,y_plot,'k',label='Batch Gradient Descent')



#################################################################################################

#2nd order Newton's method

max_ite_newton = 20;

cost_val_newton = np.zeros((max_ite_newton,1))



for i in range(max_ite_newton):

    h_x = 1/(1+np.exp(-x.dot(weight_vec)))

    cost_val_newton[i] = -(1/(np.size(x,0)))*sum(y*np.log(h_x) + (1-y)*np.log(1-h_x))

    grad = (1/(np.size(x,0)))*(x.T.dot((h_x-y)))

    # hessian matrix

    H = (1/(np.size(x,0)))*(x.T.dot(np.diag(h_x.reshape(np.size(x,0),))).dot(np.diag((1-h_x).reshape(np.size(x,0),))).dot(x))

    weight_vec = weight_vec - np.linalg.pinv(H).dot(grad)

    

y_plot = -weight_vec[0]/weight_vec[2] - (weight_vec[1]/weight_vec[2])*x_plot

plt.plot(x_plot,y_plot,'r',label='Newtons method' )

plt.xlabel('x1');plt.ylabel('x2')

plt.legend()

plt.show()
# plot cost fucntion

plt.plot(cost_val_batch,'k')

plt.title('Batch Gradient Descent');plt.xlabel('# of iterations');plt.ylabel('cost')

plt.xlim([0,100])

plt.show()



plt.plot(cost_val_newton,'b--')

plt.plot(cost_val_newton,'ro')

plt.title('2nd order Newtons method');plt.xlabel('# of iterations');plt.ylabel('cost')

plt.show()
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(solver='newton-cg')

log_reg.fit(phi,y.ravel())



y_plot = -log_reg.intercept_/log_reg.coef_[:,1] - (log_reg.coef_[:,0]/log_reg.coef_[:,1])*x_plot

plt.plot(x_plot,y_plot,'g',label='sklearn newton-cg')

y_plot = -weight_vec[0]/weight_vec[2] - (weight_vec[1]/weight_vec[2])*x_plot

plt.plot(x_plot,y_plot,'r',label='Newtons method' )

plt.scatter(c1[:,0],c1[:,1],s= 9,color='blue',marker='*',label='class1')

plt.scatter(c2[:,0],c2[:,1],s= 9,color='orange',marker='o',label='class2')

plt.xlabel('x1');plt.ylabel('x2')

plt.legend()

plt.show()