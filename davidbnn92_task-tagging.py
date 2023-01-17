import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    

from pathlib import Path



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'
skill_series = pd.Series(

    [[] for name in sorted(os.listdir(training_path))],

    index = sorted(os.listdir(training_path))

)



# 0-5

skill_series['007bbfb7.json'] = ['image_repetition', 'fractal_repetition']

skill_series['00d62c1b.json'] = ['loop_filling']

skill_series['017c7c7b.json'] = ['recoloring', 'pattern_expansion', 'pattern_repetition', 'image_expansion']

skill_series['025d127b.json'] = ['pattern_modification']

skill_series['045e512c.json'] = ['pattern_expansion', 'direction_guessing']

skill_series['0520fde7.json'] = ['detect_wall', 'separate_images', 'pattern_intersection']



# 6-10

skill_series['05269061.json'] = ['image_filling', 'pattern_expansion', 'diagonals']

skill_series['05f2a901.json'] = ['pattern_moving', 'direction_guessing', 'bring_patterns_close']

skill_series['06df4c85.json'] = ['detect_grid', 'connect_the_dots', 'grid_coloring']

skill_series['08ed6ac7.json'] = ['measure_length', 'order_numbers', 'associate_colors_to_ranks', 'recoloring']

skill_series['09629e4f.json'] = ['detect_grid', 'separate_images', 'count_tiles', 'take_minimum', 'enlarge_image', 'create_grid', 'adapt_image_to_grid']



# 11-15

skill_series['0962bcdd.json'] = ['pattern_expansion']

skill_series['0a938d79.json'] = ['direction_guessing', 'draw_line_from_point', 'pattern_expansion']

skill_series['0b148d64.json'] = ['detect_grid', 'separate_images', 'find_the_intruder', 'crop']

skill_series['0ca9ddb6.json'] = ['pattern_expansion', 'associate_patterns_to_colors']

skill_series['0d3d703e.json'] = ['associate_colors_to_colors']



# 16-20

skill_series['0dfd9992.json'] = ['image_filling', 'pattern_expansion']

skill_series['0e206a2e.json'] = ['associate_patterns_to_patterns', 'pattern_repetition', 'pattern_rotation', 'pattern_reflection', 'pattern_juxtaposition']

skill_series['10fcaaa3.json'] = ['pattern_expansion', 'image_repetition']

skill_series['11852cab.json'] = ['pattern_expansion']

skill_series['1190e5a7.json'] = ['detect_grid', 'count_hor_lines', 'count_ver_lines', 'detect_background_color', 'color_guessing', 'create_image_from_info']



# 21-25

skill_series['137eaa0f.json'] = ['pattern_juxtaposition']

skill_series['150deff5.json'] = ['pattern_coloring', 'pattern_deconstruction', 'associate_colors_to_patterns']

skill_series['178fcbfb.json'] = ['direction_guessing', 'draw_line_from_point']

skill_series['1a07d186.json'] = ['bring_patterns_close', 'find_the_intruder']

skill_series['1b2d62fb.json'] = ['detect_wall', 'separate_images', 'pattern_intersection']



# 26-30

skill_series['1b60fb0c.json'] = ['pattern_deconstruction', 'pattern_rotation', 'pattern_expansion']

skill_series['1bfc4729.json'] = ['pattern_expansion']

skill_series['1c786137.json'] = ['detect_enclosure', 'crop']

skill_series['1caeab9d.json'] = ['pattern_moving', 'pattern_alignment']

skill_series['1cf80156.json'] = ['crop']



# 31-35

skill_series['1e0a9b12.json'] = ['pattern_moving', 'gravity']

skill_series['1e32b0e9.json'] = ['detect_grid', 'separate_images', 'image_repetition', 'pattern_completion']

skill_series['1f0c79e5.json'] = ['pattern_expansion', 'diagonals', 'direction_guessing']

skill_series['1f642eb9.json'] = ['image_within_image', 'projection_unto_rectangle']

skill_series['1f85a75f.json'] = ['crop', 'find_the_intruder']



# 36-40

skill_series['1f876c06.json'] = ['connect_the_dots', 'diagonals']

skill_series['1fad071e.json'] = ['count_patterns', 'associate_images_to_numbers']

skill_series['2013d3e2.json'] = ['pattern_deconstruction', 'crop']

skill_series['2204b7a8.json'] = ['proximity_guessing', 'recoloring']

skill_series['22168020.json'] = ['pattern_expansion']



# 41-45

skill_series['22233c11.json'] = ['pattern_expansion', 'size_guessing']

skill_series['2281f1f4.json'] = ['direction_guessing', 'draw_line_from_point', 'pattern_intersection']

skill_series['228f6490.json'] = ['pattern_moving', 'loop_filling', 'shape_guessing', 'x_marks_the_spot']

skill_series['22eb0ac0.json'] = ['connect_the_dots', 'color_matching']

skill_series['234bbc79.json'] = ['recoloring', 'bring_patterns_close', 'crop']



# 46-50

skill_series['23581191.json'] = ['draw_line_from_point', 'pattern_intersection']

skill_series['239be575.json'] = ['detect_connectedness', 'associate_images_to_bools']

skill_series['23b5c85d.json'] = ['measure_area', 'take_minimum', 'crop']

skill_series['253bf280.json'] = ['connect_the_dots', 'direction_guessing']

skill_series['25d487eb.json'] = ['draw_line_from_point', 'direction_guessing', 'color_guessing']



# 51-55

skill_series['25d8a9c8.json'] = ['detect_hor_lines', 'recoloring', 'remove_noise']

skill_series['25ff71a9.json'] = ['pattern_moving']

skill_series['264363fd.json'] = ['pattern_repetition', 'pattern_juxtaposition', 'draw_line_from_point']

skill_series['272f95fa.json'] = ['detect_grid', 'mimic_pattern', 'grid_coloring']

skill_series['27a28665.json'] = ['associate_colors_to_patterns', 'take_negative', 'associate_images_to_patterns']



# 56-60

skill_series['28bf18c6.json'] = ['crop', 'pattern_repetition']

skill_series['28e73c20.json'] = ['ex_nihilo', 'mimic_pattern']

skill_series['29623171.json'] = ['detect_grid', 'separate_images', 'count_tiles', 'take_maximum', 'grid_coloring']

skill_series['29c11459.json'] = ['draw_line_from_point', 'count_tiles']

skill_series['29ec7d0e.json'] = ['image_filling', 'pattern_expansion', 'detect_grid', 'pattern_repetition']



# 61-65

skill_series['2bcee788.json'] = ['pattern_reflection', 'direction_guessing', 'image_filling', 'background_filling']

skill_series['2bee17df.json'] = ['draw_line_from_border', 'count_tiles', 'take_maximum']

skill_series['2c608aff.json'] = ['draw_line_from_point', 'projection_unto_rectangle']

skill_series['2dc579da.json'] = ['detect_grid', 'find_the_intruder', 'crop']

skill_series['2dd70a9a.json'] = ['draw_line_from_point', 'direction_guessing', 'maze']



# 66-70

skill_series['2dee498d.json'] = ['detect_repetition', 'crop', 'divide_by_n']

skill_series['31aa019c.json'] = ['find_the_intruder', 'remove_noise', 'contouring']

skill_series['321b1fc6.json'] = ['pattern_repetition', 'pattern_juxtaposition']

skill_series['32597951.json'] = ['find_the_intruder', 'recoloring']

skill_series['3345333e.json'] = ['pattern_completion', 'pattern_reflection', 'remove_noise']



# 71-75

skill_series['3428a4f5.json'] = ['detect_wall', 'separate_images', 'pattern_differences']

skill_series['3618c87e.json'] = ['gravity']

skill_series['3631a71a.json'] = ['image_filling', 'pattern_expansion', 'pattern_rotation']

skill_series['363442ee.json'] = ['detect_wall', 'pattern_repetition', 'pattern_juxtaposition']

skill_series['36d67576.json'] = ['pattern_repetition', 'pattern_juxtaposition', 'pattern_reflection', 'pattern_rotation']



# 76-80

skill_series['36fdfd69.json'] = ['recoloring', 'rectangle_guessing']

skill_series['3906de3d.json'] = ['gravity']

skill_series['39a8645d.json'] = ['count_patterns', 'take_maximum', 'crop']

skill_series['39e1d7f9.json'] = ['detect_grid', 'pattern_repetition', 'grid_coloring']

skill_series['3aa6fb7a.json'] = ['pattern_completion', 'pattern_rotation']



# 81-85

skill_series['3ac3eb23.json'] = ['draw_pattern_from_point', 'pattern_repetition']

skill_series['3af2c5a8.json'] = ['image_repetition', 'image_reflection', 'image_rotation']

skill_series['3bd67248.json'] = ['draw_line_from_border', 'diagonals','pattern_repetition']

skill_series['3bdb4ada.json'] = ['recoloring','pattern_repetition', 'holes']

skill_series['3befdf3e.json'] = ['take_negative', 'pattern_expansion']



# 86-90

skill_series['3c9b0459.json'] = ['image_rotation']

skill_series['3de23699.json'] = ['take_negative', 'crop', 'rectangle_guessing']

skill_series['3e980e27.json'] = ['pattern_repetition', 'pattern_juxtaposition', 'direction_guessing', 'pattern_reflection']

skill_series['3eda0437.json'] = ['rectangle_guessing', 'recoloring', 'measure_area', 'take_maximum']

skill_series['3f7978a0.json'] = ['crop', 'rectangle_guessing', 'find_the_intruder']



# 91-95

skill_series['40853293.json'] = ['connect_the_dots']

skill_series['4093f84a.json'] = ['gravity', 'recoloring', 'projection_unto_rectangle']

skill_series['41e4d17e.json'] = ['draw_line_from_point', 'pattern_repetition']

skill_series['4258a5f9.json'] = ['pattern_repetition', 'contouring']

skill_series['4290ef0e.json'] = ['pattern_moving', 'concentric', 'crop']



# 96-100

skill_series['42a50994.json'] = ['remove_noise', 'count_tiles']

skill_series['4347f46a.json'] = ['loop_filling', 'color_guessing']

skill_series['444801d8.json'] = ['pattern_repetition', 'pattern_expansion', 'rectangle_guessing']

skill_series['445eab21.json'] = ['measure_area', 'take_maximum']

skill_series['447fd412.json'] = ['pattern_repetition', 'draw_pattern_from_point', 'pattern_resizing']



# 101-105

skill_series['44d8ac46.json'] = ['loop_filling', 'rectangle_guessing']

skill_series['44f52bb0.json'] = ['detect_symmetry', 'associate_images_to_bools']

skill_series['4522001f.json'] = ['image_rotation', 'pairwise_analogy']

skill_series['4612dd53.json'] = ['pattern_completion', 'rectangle_guessing']

skill_series['46442a0e.json'] = ['image_repetition', 'image_reflection']



# 106-110

skill_series['469497ad.json'] = ['image_resizing', 'draw_line_from_point', 'diagonals']

skill_series['46f33fce.json'] = ['pattern_resizing', 'image_resizing']

skill_series['47c1f68c.json'] = ['detect_grid', 'find_the_intruder', 'crop', 'recolor', 'color_guessing', 'image_repetition', 'image_reflection']

skill_series['484b58aa.json'] = ['image_filling', 'pattern_expansion', 'pattern_repetition']

skill_series['48d8fb45.json'] = ['find_the_intruder', 'crop']



# 111-115

skill_series['4938f0c2.json'] = ['pattern_expansion', 'pattern_rotation', 'pattern_reflection']

skill_series['496994bd.json'] = ['pattern_reflection']

skill_series['49d1d64f.json'] = ['pattern_expansion', 'image_expansion']

skill_series['4be741c5.json'] = ['summarize']

skill_series['4c4377d9.json'] = ['image_repetition', 'image_reflection']



# 116-120

skill_series['4c5c2cf0.json'] = ['pattern_expansion', 'pattern_rotation', 'pattern_reflection']

skill_series['50846271.json'] = ['pattern_completion', 'recoloring']

skill_series['508bd3b6.json'] = ['draw_line_from_point', 'direction_guessing', 'pattern_reflection']

skill_series['50cb2852.json'] = ['holes', 'rectangle_guessing']

skill_series['5117e062.json'] = ['find_the_intruder', 'crop', 'recoloring']



# 121-125

skill_series['5168d44c.json'] = ['direction_guessing', 'recoloring', 'contouring', 'pattern_moving']

skill_series['539a4f51.json'] = ['pattern_expansion', 'image_expansion']

skill_series['53b68214.json'] = ['pattern_expansion', 'image_expansion']

skill_series['543a7ed5.json'] = ['contouring', 'loop_filling']

skill_series['54d82841.json'] = ['pattern_expansion', 'gravity']



# 126-130

skill_series['54d9e175.json'] = ['detect_grid', 'separate_images', 'associate_images_to_images']

skill_series['5521c0d9.json'] = ['pattern_moving', 'measure_length']

skill_series['5582e5ca.json'] = ['count_tiles', 'dominant_color']

skill_series['5614dbcf.json'] = ['remove_noise', 'image_resizing']

skill_series['56dc2b01.json'] = ['gravity', 'direction_guessing', 'pattern_expansion']



# 131-135

skill_series['56ff96f3.json'] = ['pattern_completion', 'rectangle_guessing']

skill_series['57aa92db.json'] = ['draw_pattern_from_point', 'pattern_repetition', 'pattern_resizing']

skill_series['5ad4f10b.json'] = ['color_guessing', 'remove_noise', 'recoloring', 'crop', 'image_resizing']

skill_series['5bd6f4ac.json'] = ['rectangle_guessing', 'crop']

skill_series['5c0a986e.json'] = ['draw_line_from_point', 'diagonals', 'direction_guessing']



# 136-140

skill_series['5c2c9af4.json'] = ['rectangle_guessing', 'pattern_expansion']

skill_series['5daaa586.json'] = ['detect_grid', 'crop', 'draw_line_from_point', 'direction_guessing']

skill_series['60b61512.json'] = ['pattern_completion']

skill_series['6150a2bd.json'] = ['image_rotation']

skill_series['623ea044.json'] = ['draw_line_from_point', 'diagonals']



# 141-145

skill_series['62c24649.json'] = ['image_repetition', 'image_reflection', 'image_rotation']

skill_series['63613498.json'] = ['recoloring', 'compare_image', 'detect_wall']

skill_series['6430c8c4.json'] = ['detect_wall', 'separate_images', 'take_complement', 'pattern_intersection']

skill_series['6455b5f5.json'] = ['measure_area', 'take_maximum', 'take_minimum', 'loop_filling', 'associate_colors_to_ranks']

skill_series['662c240a.json'] = ['separate_images', 'detect_symmetry',  'find_the_intruder', 'crop']



# 146-150

skill_series['67385a82.json'] = ['recoloring', 'measure_area', 'associate_colors_to_bools']

skill_series['673ef223.json'] = ['recoloring', 'draw_line_from_point', 'portals']

skill_series['6773b310.json'] = ['detect_grid', 'separate_images', 'count_tiles', 'associate_colors_to_numbers']

skill_series['67a3c6ac.json'] = ['image_reflection']

skill_series['67a423a3.json'] = ['pattern_intersection', 'contouring']



# 151-155

skill_series['67e8384a.json'] = ['image_repetition', 'image_reflection', 'image_rotation']

skill_series['681b3aeb.json'] = ['pattern_moving', 'jigsaw', 'crop', 'bring_patterns_close']

skill_series['6855a6e4.json'] = ['pattern_moving', 'direction_guessing', 'x_marks_the_spot']

skill_series['68b16354.json'] = ['image_reflection']

skill_series['694f12f3.json'] = ['rectangle_guessing', 'loop_filling', 'measure_area', 'associate_colors_to_ranks']



# 156-160

skill_series['6a1e5592.json'] = ['pattern_moving', 'jigsaw', 'recoloring']

skill_series['6aa20dc0.json'] = ['pattern_repetition', 'pattern_juxtaposition', 'pattern_resizing']

skill_series['6b9890af.json'] = ['pattern_moving', 'pattern_resizing', 'crop', 'x_marks_the_spot']

skill_series['6c434453.json'] = ['replace_pattern']

skill_series['6cdd2623.json'] = ['connect_the_dots', 'find_the_intruder', 'remove_noise']



# 161-165

skill_series['6cf79266.json'] = ['rectangle_guessing', 'recoloring']

skill_series['6d0160f0.json'] = ['detect_grid', 'separate_image', 'find_the_intruder', 'pattern_moving']

skill_series['6d0aefbc.json'] = ['image_repetition', 'image_reflection']

skill_series['6d58a25d.json'] = ['draw_line_from_point']

skill_series['6d75e8bb.json'] = ['rectangle_guessing', 'pattern_completion']



# 166-170

skill_series['6e02f1e3.json'] = ['count_different_colors', 'associate_images_to_numbers']

skill_series['6e19193c.json'] = ['draw_line_from_point', 'direction_guessing', 'diagonals']

skill_series['6e82a1ae.json'] = ['recoloring', 'count_tiles', 'associate_colors_to_numbers']

skill_series['6ecd11f4.json'] = ['color_palette', 'recoloring', 'pattern_resizing', 'crop']

skill_series['6f8cd79b.json'] = ['ex_nihilo', 'contouring']



# 171-175

skill_series['6fa7a44f.json'] = ['image_repetition', 'image_reflection']

skill_series['72322fa7.json'] = ['pattern_repetition', 'pattern_juxtaposition']

skill_series['72ca375d.json'] = ['find_the_intruder', 'detect_symmetry', 'crop']

skill_series['73251a56.json'] = ['image_filling', 'diagonal_symmetry']

skill_series['7447852a.json'] = ['pattern_expansion', 'pairwise_analogy']



# 176-180

skill_series['7468f01a.json'] = ['crop', 'image_reflection']

skill_series['746b3537.json'] = ['crop', 'direction_guessing']

skill_series['74dd1130.json'] = ['image_reflection', 'diagonal_symmetry']

skill_series['75b8110e.json'] = ['separate_images', 'image_juxtaposition']

skill_series['760b3cac.json'] = ['pattern_reflection', 'direction_guessing']



# 181-185

skill_series['776ffc46.json'] = ['recoloring', 'associate_colors_to_patterns', 'detect_enclosure', 'find_the_intruder']

skill_series['77fdfe62.json'] = ['recoloring', 'color_guessing', 'detect_grid', 'crop']

skill_series['780d0b14.json'] = ['detect_grid', 'summarize']

skill_series['7837ac64.json'] = ['detect_grid', 'color_guessing', 'grid_coloring', 'crop', 'extrapolate_image_from_grid']

skill_series['794b24be.json'] = ['count_tiles', 'associate_images_to_numbers']



# 186-190

skill_series['7b6016b9.json'] = ['loop_filling', 'background_filling', 'color_guessing']

skill_series['7b7f7511.json'] = ['separate_images', 'detect_repetition', 'crop']

skill_series['7c008303.json'] = ['color_palette', 'detect_grid', 'recoloring', 'color_guessing', 'separate_images', 'crop']

skill_series['7ddcd7ec.json'] = ['draw_line_from_point', 'direction_guessing', 'diagonals']

skill_series['7df24a62.json'] = ['pattern_repetition', 'pattern_rotation', 'pattern_juxtaposition', 'out_of_boundary']



# 191-195

skill_series['7e0986d6.json'] = ['color_guessing', 'remove_noise']

skill_series['7f4411dc.json'] = ['rectangle_guessing', 'remove_noise']

skill_series['7fe24cdd.json'] = ['image_repetition', 'image_rotation']

skill_series['80af3007.json'] = ['crop', 'pattern_resizing', 'image_resizing', 'fractal_repetition']

skill_series['810b9b61.json'] = ['recoloring', 'detect_closed_curves']



# 196-200

skill_series['82819916.json'] = ['pattern_repetition', 'color_guessing', 'draw_line_from_point', 'associate_colors_to_colors']

skill_series['83302e8f.json'] = ['detect_grid', 'detect_closed_curves', 'rectangle_guessing', 'associate_colors_to_bools', 'loop_filling']

skill_series['834ec97d.json'] = ['draw_line_from_border', 'pattern_repetition', 'spacing', 'measure_distance_from_side']

skill_series['8403a5d5.json'] = ['draw_line_from_point', 'pattern_repetition', 'direction_guessing']

skill_series['846bdb03.json'] = ['pattern_moving', 'pattern_reflection', 'crop', 'color_matching', 'x_marks_the_spot']



# 201-205

skill_series['855e0971.json'] = ['draw_line_from_point', 'direction_guessing', 'separate_images', 'holes']

skill_series['85c4e7cd.json'] = ['color_guessing', 'recoloring', 'color_permutation']

skill_series['868de0fa.json'] = ['loop_filling', 'color_guessing', 'measure_area', 'even_or_odd', 'associate_colors_to_bools']

skill_series['8731374e.json'] = ['rectangle_guessing', 'crop', 'draw_line_from_point']

skill_series['88a10436.json'] = ['pattern_repetition', 'pattern_juxtaposition']



# 206-210

skill_series['88a62173.json'] = ['detect_grid', 'separate_images', 'find_the_intruder', 'crop']

skill_series['890034e9.json'] = ['pattern_repetition', 'rectangle_guessing', 'contouring']

skill_series['8a004b2b.json'] = ['pattern_repetition', 'pattern_resizing', 'pattern_juxtaposition', 'rectangle_guessing', 'crop']

skill_series['8be77c9e.json'] = ['image_repetition', 'image_reflection']

skill_series['8d5021e8.json'] = ['image_repetition', 'image_reflection']



# 211-215

skill_series['8d510a79.json'] = ['draw_line_from_point', 'detect_wall', 'direction_guessing', 'associate_colors_to_bools']

skill_series['8e1813be.json'] = ['recoloring', 'color_guessing', 'direction_guesing' 'crop', 'image_within_image']

skill_series['8e5a5113.json'] = ['detect_wall', 'separate_images', 'image_repetition', 'image_rotation']

skill_series['8eb1be9a.json'] = ['pattern_repetition', 'image_filling']

skill_series['8efcae92.json'] = ['separate_images', 'rectangle_guessing', 'count_tiles', 'take_maximum', 'crop']



# 216-220

skill_series['8f2ea7aa.json'] = ['crop', 'fractal_repetition']

skill_series['90c28cc7.json'] = ['crop', 'rectangle_guessing', 'summarize']

skill_series['90f3ed37.json'] = ['pattern_repetition', 'recoloring']

skill_series['913fb3ed.json'] = ['contouring', 'associate_colors_to_colors']

skill_series['91413438.json'] = ['count_tiles', 'algebra', 'image_repetition']



# 221-225

skill_series['91714a58.json'] = ['find_the_intruder', 'remove_noise']

skill_series['9172f3a0.json'] = ['image_resizing']

skill_series['928ad970.json'] = ['rectangle_guessing', 'color_guessing', 'draw_rectangle']

skill_series['93b581b8.json'] = ['pattern_expansion', 'color_guessing', 'out_of_boundary']

skill_series['941d9a10.json'] = ['detect_grid', 'loop_filling', 'pairwise_analogy']



# 226-230

skill_series['94f9d214.json'] = ['separate_images', 'take_complement', 'pattern_intersection']

skill_series['952a094c.json'] = ['rectangle_guessing', 'inside_out']

skill_series['9565186b.json'] = ['separate_shapes', 'count_tiles', 'recoloring', 'take_maximum', 'associate_color_to_bools']

skill_series['95990924.json'] = ['pattern_expansion']

skill_series['963e52fc.json'] = ['image_expansion', 'pattern_expansion']



# 231-235

skill_series['97999447.json'] = ['draw_line_from_point', 'pattern_expansion']

skill_series['97a05b5b.json'] = ['pattern_moving', 'pattern_juxtaposition', 'crop', 'shape_guessing']

skill_series['98cf29f8.json'] = ['pattern_moving', 'bring_patterns_close']

skill_series['995c5fa3.json'] = ['take_complement', 'detect_wall', 'separate_images', 'associate_colors_to_images', 'summarize']

skill_series['99b1bc43.json'] = ['take_complement', 'detect_wall', 'separate_images', 'pattern_intersection']



# 236-240

skill_series['99fa7670.json'] = ['draw_line_from_point', 'pattern_expansion']

skill_series['9aec4887.json'] = ['pattern_moving', 'x_marks_the_spot', 'crop', 'recoloring', 'color_guessing']

skill_series['9af7a82c.json'] = ['separate_images', 'count_tiles', 'summarize', 'order_numbers']

skill_series['9d9215db.json'] = ['pattern_expansion', 'pattern_reflection', 'pattern_rotation']

skill_series['9dfd6313.json'] = ['image_reflection', 'diagonal_symmetry']



# 241-245

skill_series['9ecd008a.json'] = ['image_filling', 'pattern_expansion', 'pattern_reflection', 'pattern_rotation', 'crop']

skill_series['9edfc990.json'] = ['background_filling', 'holes']

skill_series['9f236235.json'] = ['detect_grid', 'summarize', 'image_reflection']

skill_series['a1570a43.json'] = ['pattern_moving', 'rectangle_guessing', 'x_marks_the_spot']

skill_series['a2fd1cf0.json'] = ['connect_the_dots']



# 246-250

skill_series['a3325580.json'] = ['separate_shapes', 'count_tiles', 'take_maximum', 'summarize', 'remove_intruders']

skill_series['a3df8b1e.json'] = ['pattern_expansion', 'draw_line_from_point', 'diagonals', 'bounce']

skill_series['a416b8f3.json'] = ['image_repetition']

skill_series['a48eeaf7.json'] = ['pattern_moving', 'bring_patterns_close', 'gravity', 'direction_guessing']

skill_series['a5313dff.json'] = ['loop_filling']



# 251-255

skill_series['a5f85a15.json'] = ['recoloring', 'pattern_modification', 'pairwise_analogy']

skill_series['a61ba2ce.json'] = ['pattern_moving', 'bring_patterns_close', 'crop', 'jigsaw']

skill_series['a61f2674.json'] = ['separate_shapes', 'count_tiles', 'take_maximum', 'take_minimum', 'recoloring', 'associate_colors_to_ranks', 'remove_intruders']

skill_series['a64e4611.json'] = ['background_filling', 'rectangle_guessing']

skill_series['a65b410d.json'] = ['pattern_expansion', 'count_tiles', 'associate_colors_to_ranks']



# 256-260

skill_series['a68b268e.json'] = ['detect_grid', 'separate_images', 'pattern_juxtaposition']

skill_series['a699fb00.json'] = ['pattern_expansion', 'connect_the_dots']

skill_series['a740d043.json'] = ['crop', 'detect_background_color', 'recoloring']

skill_series['a78176bb.json'] = ['draw_parallel_line', 'direction_guessing', 'remove_intruders']

skill_series['a79310a0.json'] = ['pattern_moving', 'recoloring', 'pairwise_analogy']



# 261-265

skill_series['a85d4709.json'] = ['separate_images', 'associate_colors_to_images', 'summarize']

skill_series['a87f7484.json'] = ['separate_images', 'find_the_intruder', 'crop']

skill_series['a8c38be5.json'] = ['pattern_moving', 'jigsaw', 'crop']

skill_series['a8d7556c.json'] = ['recoloring', 'rectangle_guessing']

skill_series['a9f96cdd.json'] = ['replace_pattern', 'out_of_boundary']



# 266-270

skill_series['aabf363d.json'] = ['recoloring', 'color_guessing', 'remove_intruders']

skill_series['aba27056.json'] = ['pattern_expansion', 'draw_line_from_point', 'diagonals']

skill_series['ac0a08a4.json'] = ['image_resizing', 'count_tiles', 'size_guessing']

skill_series['ae3edfdc.json'] = ['bring_patterns_close', 'gravity']

skill_series['ae4f1146.json'] = ['separate_images', 'count_tiles', 'crop']



# 271-275

skill_series['aedd82e4.json'] = ['recoloring', 'separate_shapes', 'count_tiles', 'take_minimum', 'associate_colors_to_bools']

skill_series['af902bf9.json'] = ['ex_nihilo', 'x_marks_the_spot']

skill_series['b0c4d837.json'] = ['measure_length', 'associate_images_to_numbers']

skill_series['b190f7f5.json'] = ['separate_images', 'image_expasion', 'color_palette', 'image_resizing', 'replace_pattern']

skill_series['b1948b0a.json'] = ['recoloring', 'associate_colors_to_colors']



# 276-280

skill_series['b230c067.json'] = ['recoloring', 'separate_shapes', 'find_the_intruder', 'associate_colors_to_bools']

skill_series['b27ca6d3.json'] = ['find_the_intruder', 'count_tiles', 'contouring']

skill_series['b2862040.json'] = ['recoloring', 'detect_closed_curves', 'associate_colors_to_bools']

skill_series['b527c5c6.json'] = ['pattern_expansion', 'draw_line_from_point', 'contouring', 'direction_guessing', 'size_guessing']

skill_series['b548a754.json'] = ['pattern_expansion', 'pattern_modification', 'x_marks_the_spot']



# 281-285

skill_series['b60334d2.json'] = ['replace_pattern']

skill_series['b6afb2da.json'] = ['recoloring', 'replace_pattern', 'rectangle_guessing']

skill_series['b7249182.json'] = ['pattern_expansion']

skill_series['b775ac94.json'] = ['pattern_expansion', 'pattern_repetition', 'recoloring', 'pattern_rotation', 'pattern_reflection', 'direction_guessing', 'pattern_juxtaposition']

skill_series['b782dc8a.json'] = ['pattern_expansion', 'maze']



# 286-290

skill_series['b8825c91.json'] = ['pattern_completion', 'pattern_rotation', 'pattern_reflection']

skill_series['b8cdaf2b.json'] = ['pattern_expansion', 'draw_line_from_point', 'diagonals', 'pairwise_analogy']

skill_series['b91ae062.json'] = ['image_resizing', 'size_guessing', 'count_different_colors']

skill_series['b94a9452.json'] = ['crop', 'take_negative']

skill_series['b9b7f026.json'] = ['find_the_intruder', 'summarize']



# 291-295

skill_series['ba26e723.json'] = ['pattern_modification', 'pairwise_analogy', 'recoloring']

skill_series['ba97ae07.json'] = ['pattern_modification', 'pairwise_analogy', 'rettangle_guessing', 'recoloring']

skill_series['bb43febb.json'] = ['loop_filling', 'rettangle_guessing']

skill_series['bbc9ae5d.json'] = ['pattern_expansion', 'image_expansion']

skill_series['bc1d5164.json'] = ['pattern_moving', 'pattern_juxtaposition', 'crop', 'pairwise_analogy']



# 296-300

skill_series['bd4472b8.json'] = ['detect_wall', 'pattern_expansion', 'ex_nihilo', 'color_guessing', 'color_palette']

skill_series['bda2d7a6.json'] = ['recoloring', 'pairwise_analogy', 'pattern_modification', 'color_permutation']

skill_series['bdad9b1f.json'] = ['draw_line_from_point', 'direction_guessing', 'recoloring', 'take_intersection']

skill_series['be94b721.json'] = ['separate_shapes', 'count_tiles', 'take_maximum', 'crop']

skill_series['beb8660c.json'] = ['pattern_moving', 'count_tiles', 'order_numbers']



# 301-305

skill_series['c0f76784.json'] = ['loop_filling', 'measure_area', 'associate_colors_to_numbers']

skill_series['c1d99e64.json'] = ['draw_line_from_border', 'detect_grid']

skill_series['c3e719e8.json'] = ['image_repetition', 'image_expansion', 'count_different_colors', 'take_maximum']

skill_series['c3f564a4.json'] = ['pattern_expansion', 'image_filling']

skill_series['c444b776.json'] = ['detect_grid', 'separate_images', 'find_the_intruder', 'image_repetition']



# 306-310

skill_series['c59eb873.json'] = ['image_resizing']

skill_series['c8cbb738.json'] = ['pattern_moving', 'jigsaw', 'crop']

skill_series['c8f0f002.json'] = ['recoloring', 'associate_colors_to_colors']

skill_series['c909285e.json'] = ['find_the_intruder', 'crop', 'rectangle_guessing']

skill_series['c9e6f938.json'] = ['image_repetition', 'image_reflection']



# 311-315

skill_series['c9f8e694.json'] = ['recoloring', 'pattern_repetition', 'color_palette']

skill_series['caa06a1f.json'] = ['pattern_expansion', 'image_filling']

skill_series['cbded52d.json'] = ['detect_grid', 'separate_images', 'pattern_modification', 'pattern_repetition', 'pattern_juxtaposition', 'connect_the_dots']

skill_series['cce03e0d.json'] = ['image_repetition', 'image_expansion', 'pairwise_analogy']

skill_series['cdecee7f.json'] = ['summarize', 'pairwise_analogy']



# 316-320

skill_series['ce22a75a.json'] = ['replace_pattern']

skill_series['ce4f8723.json'] = ['detect_wall', 'separate_images', 'take_complement', 'take_intersection']

skill_series['ce602527.json'] = ['crop', 'size_guessing', 'shape_guessing', 'find_the_intruder', 'remove_intruder']

skill_series['ce9e57f2.json'] = ['recoloring', 'count_tiles', 'take_half']

skill_series['cf98881b.json'] = ['detect_wall', 'separate_images', 'pattern_juxtaposition']



# 321-325

skill_series['d037b0a7.json'] = ['pattern_expansion', 'draw_line_from_point']

skill_series['d06dbe63.json'] = ['pattern_expansion', 'pairwise_analogy']

skill_series['d07ae81c.json'] = ['draw_line_from_point', 'diagonals', 'color_guessing']

skill_series['d0f5fe59.json'] = ['separate_shapes', 'count_shapes', 'associate_images_to_numbers', 'pairwise_analogy']

skill_series['d10ecb37.json'] = ['crop']



# 326-330

skill_series['d13f3404.json'] = ['image_expansion', 'draw_line_from_point', 'diagonals']

skill_series['d22278a0.json'] = ['pattern_expansion', 'pairwise_analogy']

skill_series['d23f8c26.json'] = ['crop', 'image_expansion']

skill_series['d2abd087.json'] = ['separate_shapes', 'count_tiles', 'associate_colors_to_numbers', 'recoloring']

skill_series['d364b489.json'] = ['pattern_expansion']



# 331-335

skill_series['d406998b.json'] = ['recoloring', 'one_yes_one_no', 'cylindrical']

skill_series['d43fd935.json'] = ['draw_line_from_point', 'direction_guessing', 'projection_unto_rectangle']

skill_series['d4469b4b.json'] = ['dominant_color', 'associate_images_to_colors']

skill_series['d4a91cb9.json'] = ['connect_the_dots', 'direction_guessing']

skill_series['d4f3cd78.json'] = ['rectangle_guessing', 'recoloring', 'draw_line_from_point']



# 336-340

skill_series['d511f180.json'] = ['associate_colors_to_colors']

skill_series['d5d6de2d.json'] = ['loop_filling', 'replace_pattern', 'remove_intruders']

skill_series['d631b094.json'] = ['count_tiles', 'dominant_color', 'summarize']

skill_series['d687bc17.json'] = ['bring_patterns_close', 'gravity', 'direction_guessing', 'find_the_intruder', 'remove_intruders']

skill_series['d6ad076f.json'] = ['bridges', 'connect_the_dots', 'draw_line_from_point']



# 341-345

skill_series['d89b689b.json'] = ['pattern_juxtaposition', 'summarize', 'direction_guessing']

skill_series['d8c310e9.json'] = ['pattern_expansion', 'pattern_repetition', 'pattern_completion']

skill_series['d90796e8.json'] = ['replace_pattern']

skill_series['d9f24cd1.json'] = ['draw_line_from_point', 'gravity', 'obstacles']

skill_series['d9fac9be.json'] = ['find_the_intruder', 'summarize', 'x_marks_the_spot']



# 346-350

skill_series['dae9d2b5.json'] = ['pattern_juxtaposition', 'separate_images', 'recoloring']

skill_series['db3e9e38.json'] = ['pattern_expansion', 'out_of_boundary']

skill_series['db93a21d.json'] = ['contouring', 'draw_line_from_point', 'measure_area', 'measure_length', 'algebra']

skill_series['dbc1a6ce.json'] = ['connect_the_dots']

skill_series['dc0a314f.json'] = ['pattern_completion', 'crop']



# 351-355

skill_series['dc1df850.json'] = ['contouring', 'pattern_expansion', 'out_of_boundary']

skill_series['dc433765.json'] = ['pattern_moving', 'direction_guessing', 'only_one']

skill_series['ddf7fa4f.json'] = ['color_palette', 'recoloring']

skill_series['de1cd16c.json'] = ['separate_images', 'count_tiles', 'take_maximum', 'summarize']

skill_series['ded97339.json'] = ['connect_the_dots']



# 356-360

skill_series['e179c5f4.json'] = ['pattern_expansion', 'bouncing']

skill_series['e21d9049.json'] = ['pattern_expansion', 'draw_line_from_point', 'color_palette']

skill_series['e26a3af2.json'] = ['remove_noise', 'separate_images']

skill_series['e3497940.json'] = ['detect_wall', 'separate_images', 'image_reflection', 'image_juxtaposition']

skill_series['e40b9e2f.json'] = ['pattern_expansion', 'pattern_reflection', 'pattern_rotation']



# 361-365

skill_series['e48d4e1a.json'] = ['count_tiles', 'pattern_moving', 'detect_grid', 'out_of_boundary']

skill_series['e5062a87.json'] = ['pattern_repetition', 'pattern_juxtaposition']

skill_series['e509e548.json'] = ['recoloring', 'associate_colors_to_shapes', 'homeomorphism']

skill_series['e50d258f.json'] = ['separate_images', 'detect_background_color', 'crop', 'count_tiles', 'take_maximum']

skill_series['e6721834.json'] = ['pattern_moving', 'pattern_juxtaposition', 'crop']



# 366-370

skill_series['e73095fd.json'] = ['loop_filling', 'rectangle_guessing']

skill_series['e76a88a6.json'] = ['pattern_repetition', 'pattern_juxtaposition']

skill_series['e8593010.json'] = ['holes', 'count_tiles', 'loop_filling', 'associate_colors_to_numbers']

skill_series['e8dc4411.json'] = ['pattern_expansion', 'direction_guessing']

skill_series['e9614598.json'] = ['pattern_expansion', 'direction_guessing', 'measure_length']



# 371-375

skill_series['e98196ab.json'] = ['detect_wall', 'separate_images', 'image_juxtaposition']

skill_series['e9afcf9a.json'] = ['pattern_modification']

skill_series['ea32f347.json'] = ['separate_shapes', 'count_tiles', 'recoloring', 'associate_colors_to_ranks']

skill_series['ea786f4a.json'] = ['pattern_modification', 'draw_line_from_point', 'diagonals']

skill_series['eb281b96.json'] = ['image_repetition', 'image_reflection']



# 376-380

skill_series['eb5a1d5d.json'] = ['summarize']

skill_series['ec883f72.json'] = ['pattern_expansion', 'draw_line_from_point', 'diagonals']

skill_series['ecdecbb3.json'] = ['pattern_modification', 'draw_line_from_point']

skill_series['ed36ccf7.json'] = ['image_rotation']

skill_series['ef135b50.json'] = ['draw_line_from_point', 'bridges', 'connect_the_dots']



# 381-385

skill_series['f15e1fac.json'] = ['draw_line_from_point', 'gravity', 'obstacles', 'direction_guessing']

skill_series['f1cefba8.json'] = ['draw_line_from_point', 'pattern_modification']

skill_series['f25fbde4.json'] = ['crop', 'image_resizing']

skill_series['f25ffba3.json'] = ['pattern_repetition', 'pattern_reflection']

skill_series['f2829549.json'] = ['detect_wall', 'separate_images', 'take_complement', 'pattern_intersection']



# 386-390

skill_series['f35d900a.json'] = ['pattern_expansion']

skill_series['f5b8619d.json'] = ['pattern_expansion', 'draw_line_from_point', 'image_repetition']

skill_series['f76d97a5.json'] = ['take_negative', 'recoloring', 'associate_colors_to_colors']

skill_series['f8a8fe49.json'] = ['pattern_moving', 'pattern_reflection']

skill_series['f8b3ba0a.json'] = ['detect_grid', 'find_the_intruder', 'dominant_color', 'count_tiles', 'summarize', 'order_numbers']



# 391-395

skill_series['f8c80d96.json'] = ['pattern_expansion', 'background_filling']

skill_series['f8ff0b80.json'] = ['separate_shapes', 'count_tiles', 'summarize', 'order_numbers']

skill_series['f9012d9b.json'] = ['pattern_expansion', 'pattern_completion', 'crop']

skill_series['fafffa47.json'] = ['separate_images', 'take_complement', 'pattern_intersection']

skill_series['fcb5c309.json'] = ['rectangle_guessing', 'separate_images', 'count_tiles', 'take_maximum', 'crop', 'recoloring']



# 396-399

skill_series['fcc82909.json'] = ['pattern_expansion', 'separate_images', 'count_different_colors']

skill_series['feca6190.json'] = ['pattern_expansion', 'image_expansion', 'draw_line_from_point', 'diagonals']

skill_series['ff28f65a.json'] = ['count_shapes', 'associate_images_to_numbers']

skill_series['ff805c23.json'] = ['pattern_expansion', 'pattern_completion', 'crop']



skill_series.head(10)
from sklearn.preprocessing import MultiLabelBinarizer



mlb = MultiLabelBinarizer()

skill_df = pd.DataFrame(mlb.fit_transform(skill_series), columns=mlb.classes_)

skill_df.head(10)
print("{} different tags were used.".format(len(skill_df.columns)))



barplot_df = pd.DataFrame({

    'skills': [skill for skill in skill_df.columns],

    'count': [skill_df[skill].sum() for skill in skill_df.columns]

})

plt.figure(figsize=(20, 20))

sns.barplot(x="count", y="skills", data=barplot_df.sort_values(by='count', ascending=False))

plt.title('most_used_skills')

plt.tight_layout()

plt.show()
def count_pair(skill_a, skill_b):

    if skill_a != skill_b:

        intersection = skill_df[skill_a] * skill_df[skill_b]

        union = 1 - (1 - skill_df[skill_a]) * (1 - skill_df[skill_b])

        return intersection.sum() / union.sum()

    else:

        return 0



mat = pd.DataFrame({col:[count_pair(col, col_b) for col_b in skill_df.columns] for col in skill_df.columns}, index=skill_df.columns)



plt.subplots(figsize=(25,25))

sns.heatmap(mat, square=True, cmap="YlGnBu")
count_tags = pd.DataFrame({

    'task': skill_series.index,

    'count': skill_series.apply(len).values

})



print("On average, {} tags were used per task.".format(count_tags['count'].mean()))



barplot_df = pd.DataFrame({

    'number of tags': count_tags.groupby('count').count().index,

    'count': count_tags.groupby('count').count().task

})



plt.figure(figsize=(10, 10))

sns.barplot(x="number of tags", y="count", data=barplot_df)

plt.title('Number of tags per task')

plt.tight_layout()

plt.show()
def create_df(folder_path):

    task_names_list = sorted(os.listdir(folder_path))

    task_list = []

    for task_name in task_names_list: 

        task_file = str(folder_path / task_name)

        with open(task_file, 'r') as f:

            task = json.load(f)

            task_list.append(task)

    

    df = pd.DataFrame()

    df['task_name'] = task_names_list

    df['task'] = task_list

    df['number_of_train_pairs'] = df['task'].apply(lambda x: len(x['train']))

    df['number_of_test_pairs'] = df['task'].apply(lambda x: len(x['test']))

    

    # Compare image sizes

    df['inputs_all_have_same_height'] = df['task'].apply(

        lambda task: int(len(set([len(example['input']) for example in task['train']+task.get('test')])) == 1)

    )

    df['inputs_all_have_same_width'] = df['task'].apply(

        lambda task: int(len(set([len(example['input'][0]) for example in task['train']+task.get('test')])) == 1)

    )

    df['inputs_all_have_same_shape'] = df['inputs_all_have_same_height'] * df['inputs_all_have_same_width']

    df['input_height_if_constant'] = df['task'].apply(

        lambda task: len(task['train'][0]['input'])

                     if (len(set([len(example['input']) for example in task['train']+task.get('test')])) == 1)

                     else np.nan

    )

    df['input_width_if_constant'] = df['task'].apply(

        lambda task: len(task['train'][0]['input'][0])

                     if (len(set([len(example['input'][0]) for example in task['train']+task.get('test')])) == 1)

                     else np.nan

    )

    

    df['outputs_all_have_same_height'] = df['task'].apply(

        lambda task: int(len(set([len(example['output']) for example in task['train']+task.get('test')])) == 1)

    )

    df['outputs_all_have_same_width'] = df['task'].apply(

        lambda task: int(len(set([len(example['output'][0]) for example in task['train']+task.get('test')])) == 1)

    )

    df['outputs_all_have_same_shape'] = df['outputs_all_have_same_height'] * df['outputs_all_have_same_width']

    df['output_height_if_constant'] = df['task'].apply(

        lambda task: len(task['train'][0]['output'])

                     if (len(set([len(example['output']) for example in task['train']+task.get('test')])) == 1)

                     else np.nan

    )

    df['output_width_if_constant'] = df['task'].apply(

        lambda task: len(task['train'][0]['output'][0])

                     if (len(set([len(example['output'][0]) for example in task['train']+task.get('test')])) == 1)

                     else np.nan

    )    

    

    df['in_each_pair_shape_doesnt_change'] = df['task'].apply(

        lambda task: np.prod([int(len(example['input'][0])==len(example['output'][0])

                                  and len(example['input'])==len(example['output'])

                                 ) for example in task['train']+task.get('test')

                            ])

    )

    df['in_each_pair_shape_ratio_is_the_same'] = df['task'].apply(

        lambda task: (len(set([len(example['input'][0]) / len(example['output'][0])

                                 for example in task['train']+task.get('test')]))==1) * (

                      len(set([len(example['input']) / len(example['output'])

                                 for example in task['train']+task.get('test')]))==1)

    )

    df['o/i_height_ratio_if_constant'] = df['task'].apply(

        lambda task: len(task['train'][0]['output']) / len(task['train'][0]['input'])

                     if (len(set([len(example['input']) / len(example['output'])

                                 for example in task['train']+task.get('test')]))==1)

                     else np.nan

    )

    df['o/i_width_ratio_if_constant'] = df['task'].apply(

        lambda task: len(task['train'][0]['output'][0]) / len(task['train'][0]['input'][0])

                     if (len(set([len(example['input'][0]) / len(example['output'][0])

                                 for example in task['train']+task.get('test')]))==1)

                     else np.nan

    )

    

    return df





training_descriptive_df = create_df(training_path)

training_descriptive_df.head()
df = training_descriptive_df.join(skill_df)

corrmat = df.corr()

plt.subplots(figsize=(20,20))

sns.heatmap(corrmat, vmax=0.9, square=True)
output_df = pd.DataFrame()

output_df['task_name'] = sorted(os.listdir(training_path))

output_df['task'] = training_descriptive_df['task']

output_df = output_df.join(skill_df)

output_df.to_csv('training_tasks_tagged.csv')
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



skills = [skill for skill in skill_df.columns if skill_df[skill].sum() >= 30]



for skill in skills:

    print('Evaluating skill "{}".'.format(skill))

    count_positives = skill_df[skill].sum()

    print('Number of tasks including this skill: {} (over 400 skills).'.format(count_positives))

    clf = DecisionTreeClassifier(random_state=1728)

    X = training_descriptive_df.drop(['task_name', 'task'], axis=1).fillna(0)

    y = skill_df[skill]

    

    if count_positives > 50:

        folds = 4

    else:

        folds = 2

    print('Cross-validating with {} folds.'.format(folds))

    scores = cross_val_score(clf, X, y, scoring='roc_auc', n_jobs=-1, cv=folds, verbose=0)

    print('Cross-validation mean roc-auc score {0:.2f}%, std {1:.2f}.\n'.format(np.mean(scores)*100, np.std(scores)))
def plot_one(task, ax, i,train_or_test,input_or_output):

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    

    input_matrix = task[train_or_test][i][input_or_output]

    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    

    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])

    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_title(train_or_test + ' '+input_or_output)

    





def plot_task(task):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """    

    num_train = len(task['train'])

    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))

    for i in range(num_train):     

        plot_one(task, axs[0,i],i,'train','input')

        plot_one(task, axs[1,i],i,'train','output')        

    plt.tight_layout()

    plt.show()        

        

    num_test = len(task['test'])

    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))

    if num_test==1: 

        plot_one(task, axs[0],0,'test','input')

        plot_one(task, axs[1],0,'test','output')     

    else:

        for i in range(num_test):      

            plot_one(task, axs[0,i],i,'test','input')

            plot_one(task, axs[1,i],i,'test','output')  

    plt.tight_layout()

    plt.show() 
import ipywidgets as widgets



skill_list = [skill for skill in skill_df.columns]



toggle = widgets.ToggleButton(description='Show/Delete')

out = widgets.Output(layout=widgets.Layout(border = '1px solid black'))



a = widgets.Dropdown(

    options=skill_list,

    value=skill_list[0],

    description='Skill #1:',

)

display(a)



b = widgets.Dropdown(

    options=skill_list + [None],

    value=None,

    description='Skill #2:',

)

display(b)



def f(obj):

    with out:

        if obj['new']:  

            tag_a = a.value

            tag_b = b.value

            if tag_b:

                relevant_tasks = [task_no for task_no in output_df.index

                                  if output_df.iloc[task_no][tag_a]

                                  and output_df.iloc[task_no][tag_b]

                                 ]

            else:

                relevant_tasks = [task_no for task_no in output_df.index

                                  if output_df.iloc[task_no][tag_a]

                                 ]

            if relevant_tasks:

                for task_no in relevant_tasks:

                    print('Task number', task_no)

                    example = output_df.iloc[task_no]['task']

                    plot_task(example)

            else:

                print('No match')

        else:

            out.clear_output()



toggle.observe(f, 'value')

display(toggle)

display(out)