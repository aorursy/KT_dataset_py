import numpy as np # linear algebra
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, rfft, irfft
#from scipy.io import wavfile as wf
from IPython.display import Audio
import soundfile as sf

#Teste 1 - som com pouco ruido
#sound = '/kaggle/input/heartbeat-sounds/set_a/normal__201106210943.wav'
#Teste 2 - ok
#sound = '/kaggle/input/heartbeat-sounds/set_b/normal_noisynormal_105_1305033453095_C.wav'
#Teste 3 - ok
#sound = '/kaggle/input/heartbeat-sounds/set_b/normal_noisynormal_168_1307970069434_A1.wav'
#Teste 4
#sound = '/kaggle/input/heartbeat-sounds/set_b/Bunlabelledtest__128_1306344005749_C.wav'
#Teste 5 - ok
#sound = '/kaggle/input/heartbeat-sounds/set_b/normal_noisynormal_113_1306244002866_B.wav'
#Teste 6
#sound = '/kaggle/input/heartbeat-sounds/set_a/normal__201106111136.wav'
data, samplerate = sf.read(sound)


N = len(data)
Time = N/samplerate
T = 1/samplerate
print(samplerate)
print(N)
print(Time)
print(T)
print(N//2)
print(data)
plt.plot(data)
plt.legend()
plt.show()

Audio(sound)

yf = fft(data)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
print(N, N//2)
print(yf[1500], yf[1450])
plt.plot(np.abs(yf[:N//2]))
plt.grid()
plt.show()

yf[2000:-2000] = 0
#yf[-5000:] = 0

plt.plot(np.abs(yf))
plt.grid()
plt.show()

out = ifft(yf)

print(out)

plt.plot(out)
plt.grid()
plt.show()

#audio = wf.write("AudioFiltrado.wav", samplerate, out)

sf.write('AudioFiltrado.wav', out.real, samplerate)  

Audio("AudioFiltrado.wav")