# For ability to exchange this notebook with plots

%pylab inline
import scipy.constants as const

import scipy

from scipy.io import wavfile

from IPython.core.display import HTML

import os
# From a list of https://en.wikipedia.org/wiki/Piano_key_frequencies

notes = {'0': 0

    ,'e0':20.60172,'f0':21.82676,'f#0':23.12465,'g0':24.49971,'g#0':25.95654,'a0':27.50000,'a#0':29.13524

    ,'b0':30.86771,'c0':32.70320,'c#0':34.64783,'d0':36.70810,'d#0':38.89087

    ,'e1':41.20344,'f1':43.65353,'f#1':46.24930,'g1':48.99943,'g#1':51.91309,'a1':55.00000,'a#1':58.27047

    ,'b1':61.73541,'c1':65.40639,'c#1':69.29566,'d1':73.41619,'d#1':77.78175

    ,'e2':82.40689,'f2':87.30706,'f#2':92.49861,'g2':97.99886,'g#2':103.8262,'a2':110.0000,'a#2':116.5409

    ,'b2':123.4708,'c2':130.8128,'c#2':138.5913,'d2':146.8324,'d#2':155.5635

    ,'e3':164.8138,'f3':174.6141,'f#3':184.9972,'g3':195.9977,'g#3':207.6523,'a3':220.0000,'a#3':233.0819

    ,'b3':246.9417,'c3':261.6256,'c#3':277.1826,'d3':293.6648,'d#3':311.1270

    ,'e4':329.6276,'f4':349.2282,'f#4':369.9944,'g4':391.9954,'g#4':415.3047,'a4':440.0000,'a#4':466.1638

    ,'b4':493.8833,'c4':523.2511,'c#4':554.3653,'d4':587.3295,'d#4':622.2540

    ,'e5':659.2551,'f5':698.4565,'f#5':739.9888,'g5':783.9909,'g#5':830.6094,'a5':880.0000,'a#5':932.3275

    ,'b5':987.7666,'c5':1046.502,'c#5':1108.731,'d5':1174.659,'d#5':1244.508

    ,'e6':1318.510,'f6':1396.913,'f#6':1479.978,'g6':1567.982,'g#6':1661.219,'a6':1760.000,'a#6':1864.655

    ,'b6':1975.533,'c6':2093.005,'c#6':2217.461,'d6':2349.318,'d#6':2489.016

    ,'e7':2637.020,'f7':2793.826,'f#7':2959.955,'g7':3135.963,'g#7':3322.438,'a7':3520.000,'a#7':3729.310

    ,'b7':3951.066,'c7':4186.009,'c#7':4434.922,'d7':4698.636,'d#7':4978.032

}
# I modified the output of the function for esthetic purpose

def wav_Player(filepath):

    src = """

    <head><meta http-equiv="Content-Type" content="text/html; charset=utf-8">

    <title>Simple Test</title></head><body><table><tr><td><b>%s</b></td><td>

    <audio controls="controls" style="width:300px"><source src="files/%s" type="audio/wav" />

    Your browser does not support the audio element.</audio></td></tr></table></body>

    """%(filepath, filepath)

    display(HTML(src))
# sin wave experiment

rate = 44100 #44.1 khz

duration = 2 # in sec

normedsin = lambda f, t: 2**13*sin(2*pi*f*t)

time = np.linspace(0, duration, num = rate*duration)

la = lambda t: normedsin(notes['a4'],t)

time = np.linspace(0, duration, num = rate*duration)

chart_size, chart_image = figure(figsize=(15, 4)), plot(time[0:1000], la(time)[0:1000])

wavfile.write('a4_sin.wav', rate, la(time).astype(np.int16))

wav_Player("a4_sin.wav")
# Square wave experiment

wave_len_half = int(rate/notes['a4']/2) # changeble

sound_volume = 2**13 # standart for WAV files

sound_duration = int(rate * 2) # rate * seconds

sound_wave = np.array([int((i // wave_len_half % 2 - 0.5) * 2) * sound_volume for i in range(0, sound_duration)])

time = np.linspace(0, 2, num = sound_duration)

chart_size, chart_image = figure(figsize=(15, 4)), plot(time[0:1000], sound_wave[0:1000])

wavfile.write('a4_square.wav', rate, sound_wave.astype(np.int16))

wav_Player('a4_square.wav')
# triangular wave experiment

wave_len_half = int(rate/notes['a4']/2) # changeble

sound_volume = 2**13 # standart for WAV files

sound_duration = int(rate * 2) # rate * seconds

# ((i//wave_len_half%2-0.5)*2*(i % wave_len_half)-(i//wave_len_half%2)*wave_len_half + wave_len_half/2)/wave_len_half*2

sound_wave = np.array([sound_volume*2*(i//wave_len_half%2-0.5)*((i%wave_len_half)*2-wave_len_half)/wave_len_half for i in range(0, sound_duration)])

time = np.linspace(0, 2, num = sound_duration)

chart_size, chart_image = figure(figsize=(15, 4)), plot(time[0:1000], sound_wave[0:1000])

wav_file_name = 'a4_triangular.wav'

if os.path.exists(wav_file_name):

    os.remove(wav_file_name)

wavfile.write(wav_file_name, rate, sound_wave.astype(np.int16))

wav_Player(wav_file_name)
# For file actions

make_chart   = 1

save_file    = 2

display_HTML = 4

# Universal wave generator

def generate_wave(

    note = notes['a4']

    , wave_type = 'tria'                # wave_type: 'sin', 'tria', 'square'. To be invented: saw, back_saw

    , ADSR_share = [0, 10, 30, 90, 100] # ADSR: proportions for Attack, Decay, Sustain, Release

    , ADSR_level = [0, 100, 70, 60, 0]  # ADSR: volume levels for different steps of ADSR

    , duration = 1                      # duration: in seconds, float allowed

    , volume = 2**13                    # volume: in terms of WAV file main sound volume (ADSR_level is secondary)

    , bitrate = 44100                   # bitrate: values per second

    , file_actions = 0                  # binary code: 1 = make chart, 2 = save file, 4 = display HTML

    , file_name = 'some_wav'            #

):

    # ==>> Main generator ==>>

    wave_len_half = int(bitrate / note / 2)

    wave_duration = int(bitrate * duration)

    time = np.linspace(0, duration, num = wave_duration)

    if wave_type == 'tria':

        wave = np.array([volume*2*(i//wave_len_half%2-0.5)*((i%wave_len_half)*2-wave_len_half)/wave_len_half for i in range(0, wave_duration)])

    elif wave_type == '':

        wave = [0]

    elif wave_type == '':

        wave = [0]

    else:

        wave = [0]

    # ==>> ADSR applying ==>>

    for i in range(0, wave_duration):

        if i < wave_duration*ADSR_share[1]/100:

            path_done = i / (wave_duration*ADSR_share[1]/100)

            wave[i] = path_done*wave[i]*ADSR_level[1]/100

        elif i < wave_duration*ADSR_share[2]/100:

            path_done = 1 - (i - wave_duration*ADSR_share[1]/100) / (wave_duration*(ADSR_share[2] - ADSR_share[1])/100)

            wave[i] = path_done*wave[i]*(ADSR_level[1]-ADSR_level[2])/100 + wave[i]*ADSR_level[2]/100

        elif i < wave_duration*ADSR_share[3]/100:

            path_done = 1 - (i - wave_duration*ADSR_share[2]/100) / (wave_duration*(ADSR_share[3] - ADSR_share[2])/100)

            wave[i] = path_done*wave[i]*(ADSR_level[2]-ADSR_level[3])/100 + wave[i]*ADSR_level[3]/100

        else:

            path_done = (wave_duration - i)/((ADSR_share[4] - ADSR_share[3])/100*wave_duration)

            wave[i] = path_done*wave[i]*(ADSR_level[3]-ADSR_level[4])/100 + wave[i]*ADSR_level[4]/100

    # ==>> File actions ==>>

    file_name = file_name + '.wav'

    if file_actions >> 0 & 1: # make chart

        chart_size, chart_image = figure(figsize=(15, 4)), plot(time, wave) #plot(time[0:1000], wave[0:1000])

    if file_actions >> 1 & 1: # save file

        if os.path.exists(file_name):

            os.remove(file_name)

        wavfile.write(file_name, bitrate, wave.astype(np.int16))

    if file_actions >> 2 & 1: # display HTML

        wav_Player(file_name)

    return wave
test_wave = generate_wave(

    note = notes['a5']

    , ADSR_share = [0, 5, 30, 80, 100]

    , ADSR_level = [0, 100, 80, 60, 0]

    , file_actions = (make_chart + save_file + display_HTML)

    , file_name = 'ADSR_test'

)