!git clone https://github.com/facebookresearch/demucs
%cd demucs
!apt-get install ffmpeg
!pip install lameenc
!pip install musdb
!pip install museval
!pip install treetable
!python -m demucs.separate -d cpu --dl "../../input/input-songs/BeKind.mp3"
!ls separated/demucs/BeKind
from IPython.display import Audio
Audio("separated/demucs/BeKind/vocals.wav")
Audio("separated/demucs/BeKind/bass.wav")
Audio("separated/demucs/BeKind/drums.wav")
Audio("separated/demucs/BeKind/other.wav")