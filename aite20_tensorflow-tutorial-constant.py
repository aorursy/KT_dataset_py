import tensorflow as tf
a = tf.constant(7, dtype=tf.int64)
b = tf.constant(3, dtype=tf.int64)
c = tf.constant([4, 2, 1, 9])
d = tf.constant([[9, 5, 2],
                 [7, 5, 2],
                 [1, 2, 0]], dtype=tf.int64)
e = tf.constant('I love this stuff.', dtype=tf.string)

with tf.Session() as session:
    print(f"a={session.run(a)}")
    print(f"b={session.run(b)}")
    print(f"c={session.run(c)}")
    print(f"d={session.run(d)}")
    print(f"e={session.run(e)}")
    
    session.close()
with tf.Session() as session:
    print(session.run(a))
    print(session.run(b))
    print(session.run(c))
    print(session.run(d))
    print(session.run(e))
