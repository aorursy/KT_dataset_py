!pip install moviepy
from moviepy.editor import TextClip, concatenate
text_list = ["Once upon a time", "there was a king", "who wanted to become", "a magician"]

clip_list = []



for text in text_list:

    txt_clip = TextClip(text, fontsize=50, color='black', bg_color='white', size=(640, 400)).set_duration(2)

    clip_list.append(txt_clip)



final_clip = concatenate(clip_list, method="compose")

final_clip.write_videofile("test.mp4", fps=24, codec='mpeg4')
!ls