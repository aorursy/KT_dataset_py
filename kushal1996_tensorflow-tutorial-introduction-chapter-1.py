'''Importing tensorflow'''
import tensorflow as tf
'''tf.Varibale(value , name)'''
x = tf.Variable( 2 , name = "x" )
y = tf.Variable( 4 , name = "y" )

'''equation'''
f = x*x*y + y + 2
'''creating session'''
session = tf.Session()

'''initializing variables
 x = 2 and y = 4 '''
session.run(x.initializer)
session.run(y.initializer)

'''evaluating f 
 f = x*x*y + y + 2'''
result = session.run(f)
print(result)

'''Terminating the session'''
session.close()
with tf.Session() as session :
    '''initializing variables'''
    x.initializer.run()
    y.initializer.run()
    '''evaluating f'''
    result = f.eval()
    '''session closed automatically'''
    
print(result)
'''initializing a node'''
init = tf.global_variables_initializer()

with tf.Session() as session:
    '''initializing all the variables'''
    init.run()
    '''evaluating f'''
    result = f.eval()
    '''session is closed automatically'''
    
print(result)
'''Creating an Interactive session.'''
session = tf.InteractiveSession()
'''initializing the global variables'''
init.run()
'''evaluating f'''
result = f.eval()
print(result)

'''terminating session'''
session.close()