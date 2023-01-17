import numpy as np
import os
import matplotlib.pyplot as plt

arecibo = '00000010101010000000000001010000010100000001001000100010001001011001010101010101010100100100000000000000000000000000000000000001100000000000000000001101000000000000000000011010000000000000000001010100000000000000000011111000000000000000000000000000000001100001110001100001100010000000000000110010000110100011000110000110101111101111101111101111100000000000000000000000000100000000000000000100000000000000000000000000001000000000000000001111110000000000000111110000000000000000000000011000011000011100011000100000001000000000100001101000011000111001101011111011111011111011111000000000000000000000000001000000110000000001000000000001100000000000000010000011000000000011111100000110000001111100000000001100000000000001000000001000000001000001000000110000000100000001100001100000010000000000110001000011000000000000000110011000000000000011000100001100000000011000011000000100000001000000100000000100000100000001100000000100010000000011000000001000100000000010000000100000100000001000000010000000100000000000011000000000110000000011000000000100011101011000000000001000000010000000000000010000011111000000000000100001011101001011011000000100111001001111111011100001110000011011100000000010100000111011001000000101000001111110010000001010000011000000100000110110000000000000000000000000000000000011100000100000000000000111010100010101010101001110000000001010101000000000000000010100000000000000111110000000000000000111111111000000000000111000000011100000000011000000000001100000001101000000000101100000110011000000011001100001000101000001010001000010001001000100100010000000010001010001000000000000100001000010000000000001000000000100000000000000100101000000000001111001111101001111000'
arecibo = ",".join([arecibo[i:i+1] for i in range(0, len(arecibo), 1)])
arecibo = np.fromstring(arecibo, dtype=int, sep=',')

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(np.array(arecibo).reshape(73,23))
import ibmseti
path = '../input/primary_small_v3/primary_small_v3/'
primarymediumlist = os.listdir(path)
firstfile = primarymediumlist[0]
print(path + firstfile)

data_1 = ibmseti.compamp.SimCompamp(open(path + firstfile,'rb').read())
data_1.header().get("signal_classification")
spectrogram = data_1.get_spectrogram()

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(spectrogram,  aspect = spectrogram.shape[1] / spectrogram.shape[0])
def plot_spectrogram(index_num):
    file = primarymediumlist[index_num]
    data = ibmseti.compamp.SimCompamp(open(path + file,'rb').read())
    spectrogram = data.get_spectrogram()
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title(data.header().get("signal_classification"))
    ax.imshow(spectrogram,  aspect = spectrogram.shape[1] / spectrogram.shape[0])

plot_spectrogram(0) #narrowbanddrd
plot_spectrogram(2) #narrowband
plot_spectrogram(4) #squiggle
plot_spectrogram(11) #brightpixel
plot_spectrogram(13) #noise
plot_spectrogram(14) #squarepulsednarrowband
plot_spectrogram(17) #squigglesquarepulsednarrowband
plt.hist(spectrogram)
plt.title(data_1.header().get("signal_classification"))
plt.show()
plt.hist(np.log(spectrogram))
plt.title(data_1.header().get("signal_classification"))
plt.show()
complex_data = data_1.complex_data()
complex_data = complex_data.reshape(32, 6144)
complex_data = complex_data * np.hanning(complex_data.shape[1]) #This step applies something called a Hanning Window
cpfft = np.fft.fftshift( np.fft.fft(complex_data), 1)
spectrogram = np.abs(cpfft)**2

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(np.log(spectrogram),  aspect = spectrogram.shape[1] / spectrogram.shape[0])
#png_path = ''
#zip_path = ''
#zip_list = os.listdir(zip_path)

#i = 1
#for z in zip_list:
#    zz = zipfile.ZipFile(zip_path + z)
#    primarymediumlist = zz.namelist()
#    primarymediumlist.pop(0)
#    for f in primarymediumlist:
#        i=i+1
#        temp_data = ibmseti.compamp.SimCompamp(zz.open(f, 'r').read())
#        temp_type = temp_data.header().get("signal_classification")
#        temp_spectrogram = temp_data.get_spectrogram()
#        matplotlib.image.imsave(png_path + str(i) + '_' + str(temp_type) + '.png', np.log(temp_spectrogram))
#        print("zip file: " + str(z), ", file: " + str(i))