import tensorflow as tf
import numpy as np

def fx(coeffs,x):
    a0=coeffs[0]
    a1=coeffs[1]
    a2=coeffs[2]
    a3=coeffs[3]
    a4=coeffs[4]
    result=(a0  +tf.multiply(a1,tf.math.pow(x,1))
                +tf.multiply(a2,tf.math.pow(x,2))
                +tf.multiply(a3,tf.math.pow(x,3))
                +tf.multiply(a4,tf.math.pow(x,4))
             )
    return result


def fxd(coeffs,x):
    a0=coeffs[0]
    a1=coeffs[1]
    a2=coeffs[2]
    a3=coeffs[3]
    a4=coeffs[4]
    
    result=(a1  +2.0*tf.multiply(a2,tf.math.pow(x,1))
                +3.0*tf.multiply(a3,tf.math.pow(x,2))
                +4.0*tf.multiply(a4,tf.math.pow(x,3))
             )
    return result
    
  
    
    
def fxdd(coeffs,x):
    a0=coeffs[0]
    a1=coeffs[1]
    a2=coeffs[2]
    a3=coeffs[3]
    a4=coeffs[4]
    
    result=(2.0*a2
                +6.0*tf.multiply(a3,tf.math.pow(x,1))
                +12.0*tf.multiply(a4,tf.math.pow(x,2))
             )
    return result


def hx(coeffs,x):
    result=x-(
                (2.0*fx(coeffs,x)*fxd(coeffs,x))
                /
                (
                    2.0*fxd(coeffs,x)*fxd(coeffs,x)
                    -fxd(coeffs,x)*fxdd(coeffs,x)
                )
            )
    return result
    
coeffs = tf.placeholder(dtype=tf.float64, shape=(5,))
x = tf.placeholder(dtype=tf.float64)
finaloutcome = hx(coeffs, x)
with tf.Session() as sess:
    a=[-2.0,0.0,1.0,0.0,0.0]
    r=sess.run(finaloutcome,feed_dict={coeffs:a,x:2})
    print(r)
    for i in range(10):
        r=sess.run(finaloutcome,feed_dict={coeffs:a,x:r})
        print(r)

coeffs = tf.placeholder(dtype=tf.float64, shape=(5,))
x = tf.placeholder(dtype=tf.float64)
finaloutcome = hx(coeffs, x)
with tf.Session() as sess:
    a=[-27.0,0.0,0.0,1.0,0.0]
    r=sess.run(finaloutcome,feed_dict={coeffs:a,x:2})
    print(r)
    for i in range(10):
        r=sess.run(finaloutcome,feed_dict={coeffs:a,x:r})
        print(r)

