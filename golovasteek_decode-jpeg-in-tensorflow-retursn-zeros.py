import tensorflow as tf
# trying to read one image and pring result. It's the matrix of zeros

graph = tf.Graph()

with graph.as_default():

    tf_img = tf.image.decode_jpeg(tf.read_file('../input/train-jpg/train_0.jpg'))



with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()

    img = session.run([tf_img])

    

print(img)
result = run(

    'convert -colorspace sRGB ../input/train-jpg/train_0.jpg /tmp/fixed-train.jpg'.split())

print(result)

print(result.stdout)

print(result.stderr)
print(check_output(['convert', '-h']))