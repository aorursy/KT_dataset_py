import numpy as np
def Gradient_Descent(x,y):
    
    
    m_curr = 0
    b_curr = 0
    iteration = 10000
    n = len(x)
    learning_rate = 0.001
    
    
    
    for i in range(iteration):
        
        y_pred = m_curr*x + b_curr
        
        cost = 1/n * sum([val**2 for val in (y-y_pred)])
        
        m_derivative = -(2/n) * sum(x * (y-y_pred))
        b_derivative = -(2/n) * sum(y-y_pred)
        
        m_curr = m_curr - learning_rate * m_derivative
        b_curr = b_curr - learning_rate * b_derivative
        
        
        print(f"iteration {i}   m {m_curr}   b {b_curr}   cost {cost}")
        
        
        
    
    
    
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
Gradient_Descent(x,y)