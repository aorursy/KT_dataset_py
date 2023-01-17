!apt-get install ffmpeg
!pip install spleeter
from IPython.display import Audio
!cp ../input/input-songs/BeKind.mp3 audio_example.mp3
Audio('./audio_example.mp3')
!spleeter separate -h
!spleeter separate -i audio_example.mp3 -o ./
!ls ./audio_example
Audio('./audio_example/vocals.wav')
Audio('./audio_example/accompaniment.wav')
