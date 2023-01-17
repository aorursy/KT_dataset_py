%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
def check_if_exit(v):
    # Takes a vector v=(x,y)
    # Checks if v has intersected with the boundary of D
    if np.linalg.norm(v,2) >=1:
        return True
    return False
    
def simulate_exit_time(v):
    # Simulates exit time starting at v=(x,y), returns exit position
    delta_t = np.sqrt(.001)
    exit = False
    x = v.copy()
    while not exit:
        x = x + delta_t * np.random.normal(0,1,size=2)
        exit = check_if_exit(x)
    return x
    
v=np.array((0,0)) # The origin
exit_times = np.array([simulate_exit_time(v) for k in range(0,10) ])


circle1=plt.Circle((0,0),1,color='r', alpha=.5)
plt.gcf().gca().add_artist(circle1)
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.scatter(exit_times[:,0],exit_times[:,1])
np.random.seed(8) #Side Infinity
    
def check_if_exit(v):
    # Takes a vector v=(x,y)
    # Checks if v has intersected with the boundary of D
    if np.linalg.norm(v,2) >=1:
        return True
    return False
    
def simulate_exit_time(v):
    # Simulates exit time starting at v=(x,y), returns exit position
    delta_t = np.sqrt(.001)
    exit = False
        
    # Copy because simulation modifies in place
    if hasattr(v,'copy'): # For NumPy arrays
        x = v.copy() 
    else:
        x = np.array(v) # We input a non-NumPy array
    while not exit:
        x += delta_t * np.random.normal(0,1,size=2) # += modifies in place
        exit = check_if_exit(x)
    return x
        
v=np.array((.5,.5)) # The origin
u = lambda x : np.linalg.norm(x,2)*np.cos(np.arctan2(x[1],x[0]))
f = lambda x : np.cos(np.arctan2(x[1],x[0]))
    
def get_exp_f_exit(starting_point, n_trials): 
    return np.mean([f(simulate_exit_time(starting_point)) for k in range(0,n_trials)])
    
exp_f_exit = get_exp_f_exit(v,2000) # Expected value of f(Exit(x,d))
print('The value u(v) = %s\nThe value of Exp(f(Exit))=%s' %(u(v), exp_f_exit)) 
#Simulating the PDE
lin = np.linspace(-1, 1, 100)
x, y = np.meshgrid(lin, lin)
print(x.shape)
u_vec = np.zeros(x.shape)
bm_vec_10 = np.zeros(x.shape)
#bm_vec_100 = np.zeros(x.shape)
#bm_vec_1000 = np.zeros(x.shape)
    
# Convert u to a solution in x,y coordinates
u_x = lambda x,y : np.linalg.norm(np.array([x,y]),2)*np.cos(np.arctan2(y,x))
    
# Calculate actual and approximate solution for (x,y) in D
for k in range(0,x.shape[0]):
    for j in range(0,x.shape[1]):
        x_t = x[k,j]
        y_t = y[k,j]
            
        # If the point is outside the circle, the solution is undefined
        if np.sqrt((x_t)**2 + (y_t)**2) > 1:
            continue
            
        # Calculate function value at this point for each image
        u_vec[k,j] = u_x(x_t,y_t)
        bm_vec_10[k,j] = get_exp_f_exit((x_t,y_t),10)
        #bm_vec_100[k,j] =  get_exp_f_exit((x_t,y_t),100)
        #bm_vec_1000[k,j] =  get_exp_f_exit((x_t,y_t),1000)

fig = plt.figure()
ax = fig.add_subplot(121)
plt.imshow(u_vec)
plt.title('Actual Solution')
    
ax = fig.add_subplot(122)
plt.title('BM Solution (10)')
plt.imshow(bm_vec_10)
    
fig = plt.figure()
    
#ax = fig.add_subplot(121)
#plt.title('BM Solution (100)')
#plt.imshow(bm_vec_100)
    
#ax = fig.add_subplot(122)
#plt.title('BM Solution (1000)')
#plt.imshow(bm_vec_1000)

