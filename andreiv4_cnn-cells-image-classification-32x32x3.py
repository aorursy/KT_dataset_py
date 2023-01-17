import pandas as pd

import numpy as np

import imageio
df = pd.read_csv( '../input/train.csv' )
def load_images( df, folder ):



    images = np.zeros(( len( df ), 32, 32, 3 ), dtype=np.float64 )



    for i, file in enumerate( df.id ):

        images[i] = imageio.imread( folder + '/' + file )



    return ( images - 128 ) / 64



images = load_images( df, '../input/train/train' )
from keras.models import Model

from keras.layers import Dropout, MaxPooling2D, GlobalAveragePooling2D, Dense, SeparableConv2D

from keras.layers import Input, ReLU, BatchNormalization, Conv2D, Activation





def ConvCell( m, filters, kernel=3 ):

    

    m = SeparableConv2D( filters, kernel, padding='same' )( m )

    m = BatchNormalization()( m )

    m = ReLU()( m )

    

    return m



def DeepConvCell( m, n, filters, kernel=3 ):



    for _ in range( n ): m = ConvCell( m, filters, kernel )

        

    m = MaxPooling2D()( m )

    

    return m



n_inp = Input( shape=( 32, 32, 3 ) )

conv0 = DeepConvCell( n_inp, 3, 32 )

rg0   = Dropout( .4 )( conv0 )

conv1 = DeepConvCell( rg0, 3, 64 )

rg1   = Dropout( .4 )( conv1 )

conv2 = DeepConvCell( rg1, 3, 128 )





gl_avg_pool = GlobalAveragePooling2D()( conv2 )

fc = Dense( 1, activation='sigmoid' )( gl_avg_pool )



m = Model( inputs=n_inp, outputs=fc )

m.compile( loss='binary_crossentropy', optimizer='adam' )

m.summary()
from imblearn.over_sampling import RandomOverSampler



data = images.reshape( 17_500, -1 )

data, target = RandomOverSampler().fit_resample( data, df.has_cactus )

data = data.reshape( len( data ), 32, 32, 3 )
m.fit( data, target, batch_size=64, epochs=20, verbose=2 )
print( 'Loss:', s.history['loss'][-1])
import os



df2 = pd.DataFrame({ 'id': os.listdir( '../input/test/test' ), 'has_cactus': np.zeros( 4_000 )})



p = m.predict( load_images( df2, '../input/test/test' ))

p[ p <  .5 ] = 0

p[ p >= .5 ] = 1



df2.has_cactus = p.astype( np.uint8 )



df2.sample( 10 )
df2.to_csv( 'submission.csv', index=False )