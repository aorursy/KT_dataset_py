import tensorflow as tf 

tl = tf.layers
def fpn_up_mrg(x, skp_x, n_filts=256):

    with tf.name_scope('fpn_up_mrg'):

        # 1x1 convolution to the skip connection 

        skp_x = tl.conv2d(skp_x, n_filts, (1,1))

        

        shp = tf.shape(x)

        x = image.resize_nearest_neighbor(x, [shp[1]*2, shp[2]*2], True)

        x = x + skp_x

        x = tl.conv2d(x, n_filts, (3,3), padding='same')

    return x 



# c1-c4 are skip connections from feature extractor network. 

def fpn(c0,c1,c2,c3,c4):

    with tf.name_scope('fpn'):

        p4 = tl.conv2d(c4, 256, (1, 1))

        p3 = fpn_up_mrg(p4, c3)

        p2 = fpn_up_mrg(p3, c2)

        p1 = fpn_up_mrg(p2, c1) 

        p0 = fpn_up_mrg(p1, c0)

    return p0, p1, p2, p3, p4 