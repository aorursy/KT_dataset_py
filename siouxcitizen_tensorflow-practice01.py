# tensorlfowのインポート
# tensorflowバージョン確認
# import tensorflow library
# confirm tensorflow version
import tensorflow as tf
tf.__version__
#「Hello, TensorFlow!」を出力
# output "Hello, TensorFlow!"
#b?
hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()
print(sess.run(hello))
# 定数 a と定数 b の合計を出力
# output sum of constant a and constant b
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))