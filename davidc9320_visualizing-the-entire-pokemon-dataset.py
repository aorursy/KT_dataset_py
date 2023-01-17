import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image, ImageDraw, ImageFont
%matplotlib inline

import matplotlib.pyplot as plt
import operator
from tqdm import tqdm

import os
print(os.listdir('..'))
print(os.listdir("../input"))
df = pd.read_csv('../input/pokemon-sun-and-moon-gen-7-stats/pokemon.csv')
df.sample(10)
def correctError(column, row, new_value):
    print(df.loc[row, 'forme'], column, ':\t', df.loc[row, column], end=' --> ')
    df.loc[row, column] = new_value
    print(df.loc[row, column])
    
correctError('pre-evolution', 267, 'Wurmple')
correctError('pre-evolution', 731, 'Pikipek')
correctError('pre-evolution', 757, 'Salandit')
correctError('forme', 810, 'Raticate (Alola Form)')
correctError('type2', 837, 'Flying')
def matchSameFormes(x):
    #print(x['forme'])
    match = df[df['ndex'] ==  x['ndex']]
    #print(match, len(match))
    if len(match)>1:
        if 'Alola' in x['forme']:
            return 1        
        min_id = match['id'].min()
        feats_list = ['hp', 'attack', 'defense', 'spattack', 'spdefense', 'speed']
        for f in feats_list:
            feats_match = match[f] == x[f]
            def dummyFun(x):
                if x == True:
                    return 1
                else:
                    return 0
                
            c = feats_match.apply(dummyFun).sum()
            if c < 2:
                return 1
        
        if x['id'] == min_id:
            return 1
        else:
            return 0
    else:
        
        #for f in feats_list:   
        return 1

df['keep'] = df.apply(matchSameFormes, axis=1)
df = df[df['keep']==1]
def findPreEvolutionIdx(x):
    #print(x)
    #print(pre_evo, end='\t')
    if not pd.isnull(x):
        #match name to pokemon
        try:
            match = df[df['forme'] == x]
            if len(match) == 0:
                match = df[df['species'] == x]
            #    pre_evo = pre_evo.split(' ')
            #    pre_evo = pre_evo[0]
            #    match = df[d['species'] == pre_evo]
            match = match.iloc[0]
            return match['ndex']
        except:
            print(x)
            #print(pre_evo)
            #print(len(df_2[df_2['species'] == pre_evo].iloc[0]))
            #print(df_2[df_2['species'] == pre_evo])
            raise Exception('Error')
    else:
        #print('no match')
        return -1

df['pre-evolution'] = df['pre-evolution'].apply(findPreEvolutionIdx)
x = df.iloc[2]
def matchEvoFamily(x):
    #print(x['name'].ljust(10, '-'))
    #print(type(x))
    pre_evo = x['pre-evolution']
    if pre_evo == -1:
        output = x['ndex']
        #print(output)
        return output
    else:
        pre_evo = df[df['ndex'] == pre_evo].iloc[0]
        #print(pre_evo['name'])
        output = matchEvoFamily(pre_evo)
        #print(output)
        return output 

df['evo_family'] = df.apply(matchEvoFamily, axis=1)
df[['id', 'ndex', 'species', 'forme', 'keep', 'pre-evolution', 'evo_family']].sample(10)
other_df = pd.read_csv('../input/pokemon/pokemon.csv')
other_df.sample(5)

def isLegendary(x):
    match = other_df[other_df['pokedex_number'] == x]
    if len(match>0):
        if match.iloc[0]['is_legendary'] == 1:
            return 1
    return 0

df['is_legendary'] = df['ndex'].apply(isLegendary)
correctError('is_legendary', 801, 1)
def countStage(x, counter=0):
    if x['is_legendary'] == 1:
        return 3
    if 'Mega ' in x['forme']:
        #print(x['forme'])
        return 3
        
    pre_evo = x['pre-evolution']
    if pre_evo == -1:
        output = counter
        #print(output)
        return output
    else:
        pre_evo = df[df['ndex'] == pre_evo].iloc[0]
        #print(pre_evo['name'])
        counter +=1
        output = countStage(pre_evo, counter)
        #print(output)
        return output 

df['stage'] = df.apply(countStage, axis=1)
df[['species', 'forme', 'stage']].sample(15)
print(df.columns)
stats_list = ['hp', 'attack', 'defense', 'spattack', 'spdefense', 'speed']
print(stats_list)
for el in stats_list:
    if el in df.columns:
        print(1, end='\t')
    else:
        print(0, end='\t')
df['total'] = df[stats_list].sum(axis = 1)
df.sample(3)
max_stats = {}
min_stats = {}
for stat in stats_list:
    max_stats[stat] = df[stat].max()
    min_stats[stat] = df[stat].min()
print(max_stats)
print(min_stats)
max_total = df['total'].max()
print(max_total)
df.type1.unique()
type_color = {
    'normal' : (168, 167, 122),
    'fire' : (238, 129, 48),
    'water' : (99, 144, 240),
    'electric' : (247, 208, 44),
    'grass' : (122, 199, 76),
    'ice' : (150, 217, 214),
    'fighting': (194, 46, 40),
    'poison' : (163, 62, 161),
    'ground' : (226, 191, 101),
    'flying' : (169, 143, 243),
    'psychic' : (249, 85, 135),
    'bug' : (166, 185, 26),
    'rock' : (182, 161, 54),
    'ghost': (115, 87, 151),
    'dragon' : (111, 53, 252),
    'dark' : (112, 87, 70),
    'steel' : (183, 183, 206),
    'fairy' : (214, 133, 173)
}

im = Image.new('RGBA', (200, 20*len(type_color)))
draw = ImageDraw.Draw(im)
i = 0
for t in type_color.keys():
    draw.rectangle([0, 20*i, 200, 20*(i+1)], fill=type_color[t])
    draw.text([5, 20*i + 5], t)
    i += 1
im
def defineStatsColors(start_color = (70, 120, 100, 255), final_color = (210, 255, 230, 255)):
    color_diff = tuple(map(operator.sub, final_color, start_color))
    N = len(max_stats)
    color_step_tmp = tuple(map(operator.truediv, color_diff, (N, N, N, N)))
    color_step = ()
    for el in color_step_tmp:
        color_step += (round(el), )
    #print(color_step)
    colors = []
    for i in range(0, N):
        colors.append(tuple(map(operator.add, start_color, tuple(map(operator.mul, color_step, (i, i, i, i))))))
    #print(colors)
    return colors
def drawStats(draw, features, max_stats, x_offset = 0, y_offset = 0, max_len_segm = 100, H=20):
    x1 = x_offset    
    colors = defineStatsColors()
    for s, color in zip(stats_list, colors):    
        x0 = x1 + 3
        y0 = y_offset
        x1 += features[s]/max_stats[s] * max_len_segm
        y1 = y0 + H
        bounding_box = [x0, y0, x1, y1]
        #print(bounding_box)
        draw.rectangle(bounding_box, fill=color)
        

with Image.new('RGBA', (600, 150), (0,0,0,255)) as im:
    draw = ImageDraw.Draw(im)
    for i in range(0, 5):
        if i == 0:
            y_offset = 5
        else:
            y_offset += 20 + 5 
        #print(df.iloc[i]['name'])
        drawStats(draw, df.iloc[i], max_stats, x_offset= 5, y_offset = y_offset)
    plt.imshow(im)
def drawType(draw, features, W = 20, H = 20, x_offset = 0, y_offset = 0):
    W = 20  
    def drawTypeRect(draw, W, H, color=(255,0,0,255), x_offset=0, y_offset=0):
        x0 = x_offset
        x1 = x0 + W
        y0 = y_offset
        y1 = y0 + H
        bounding_box = [x0, y0, x1, y1]
        draw.rectangle(bounding_box, fill=color)
    
    if pd.isna(features['type2']):
        color = type_color[features['type1'].lower()]
        drawTypeRect(draw, W, H, color=color, x_offset=x_offset, y_offset=y_offset)
    else:
        for t, i in zip(['type1', 'type2'], [0,1]):
            color = type_color[features[t].lower()]
            x_offset += (W//2)*i
            drawTypeRect(draw, W//2, H, color=color, x_offset=x_offset, y_offset=y_offset)

with Image.new('RGBA', (30, 30), (0,0,0,255)) as im:
    draw = ImageDraw.Draw(im)
    drawType(draw, df.iloc[np.random.randint(len(df))], x_offset=5, y_offset=5)
    plt.imshow(im)
dfgb = df.groupby('evo_family')
groups = dfgb.groups
def drawGroup(draw, gp, x_offset_0=0, y_offset_0=0, W=20, H=20):
    #starting values
    x_offset = x_offset_0
    y_offset = y_offset_0
    for i in range(len(gp)):    
        #print('-'*30)
        
        features = gp.iloc[i]
        if i>0:
            y_offset += H + 5
            W = 20 
            H = 20 + (features['stage'])*10
        elif features['is_legendary'] == 1:
            H=60
        # text
        x_offset_text = x_offset + 4
        draw.text([x_offset_text, y_offset], text=str(features['ndex']))
        if 'Mega ' in features['forme']:
            draw.text([x_offset_text, y_offset+12], text=str('M'))
        if 'Alola' in features['forme']:
            draw.text([x_offset_text, y_offset+12], text=str('A'))
            
        x_offset_type = x_offset_text + 20
        #print('W', W, 'H', H, 'x_offset', x_offset_type, 'y_offset', y_offset)
        drawType(draw, features, W=W, H=H, x_offset=x_offset_type, y_offset=y_offset)
        x_offset_stats = x_offset_type + W + 7
        drawStats(draw, features, max_stats, x_offset=x_offset_stats, y_offset=y_offset, H=H)
        #print('W', W, 'H', H, 'x_offset', x_offset_stats, 'y_offset', y_offset)
    
    y_offset_N = y_offset+H
    draw.line([(x_offset_0, y_offset_0), (x_offset_0, y_offset_N)])
    return y_offset_N, y_offset_N-y_offset_0
H_MAX = 3550
W_MAX = 5350
im = Image.new('RGBA', (W_MAX, H_MAX), color=(25,0,25,255))
draw = ImageDraw.Draw(im)

groups = list(dfgb.groups.keys())
y_offset = y_offset_0 = 5
x_offset = 10

LEN_MAX = 0
footer_h = 999
for i in tqdm(range(len(groups))):
    gp = dfgb.get_group(groups[i])
    #print(y_offset)
    if H_MAX - y_offset < 300:
        if H_MAX - y_offset < footer_h:
            footer_h = H_MAX - y_offset
        y_offset = y_offset_0
        x_offset += 450
    y_offset, LEN = drawGroup(draw, gp, y_offset_0=y_offset, x_offset_0=x_offset) 
    y_offset += y_offset_0*4
    if LEN > LEN_MAX:
        LEN_MAX = LEN

#print(LEN_MAX)
im
def createColorLegend():
    im_color = Image.new('RGBA', (1280, 20), color=(25,0,25,255))
    draw = ImageDraw.Draw(im_color)
    #draw colors
    x_0 = 0
    for i, t in enumerate(type_color.keys()):
        W = 20
        if i>0:
            x_0 = x_1 + textsize[0] + 20
        x_1 = x_0 + W
        draw.rectangle([x_0, 0, x_1, W], fill=type_color[t])
        textsize = draw.textsize(t)
        x_1_text = x_1 + 3
    #    print(textsize)
        draw.text([x_1_text, 5], t)
        final_len = x_1_text + textsize[0]
    #print(final_len)
    return im_color
def createStatsLegend():
    im_legend = Image.new('RGBA', (950, 20), color=(25,0,25,255))
    draw = ImageDraw.Draw(im_legend)
    #draw stats
    x_0 = 0
    colors=defineStatsColors()
    for i, t in enumerate(stats_list):
        H = 20
        W = 150
        if i>0:
            x_0 = x_1 + 10
        x_1 = x_0 + W
        draw.rectangle([x_0, 0, x_1, W], fill=colors[i])
        textsize = draw.textsize(t)
        x_0_text = x_0 + 5
        #    print(textsize)
        draw.text([x_0_text, 5], t, fill=(0,0,0))
        final_len = x_1
        #print(final_len)
    return im_legend
header_h = 300
im_com = Image.new('RGBA', (W_MAX, H_MAX+header_h-footer_h+10), color=(25,0,25,255))
draw = ImageDraw.Draw(im_com)
title = 'Visualizing the entire pokedex'
im_clr_legend = createColorLegend()
im_stt_legend = createStatsLegend()
#paste image
im_com.paste(
    im_clr_legend, 
    (
        round(im_com.size[0]/2 - im_clr_legend.size[0]/2), 
        round(header_h - 100)
    )
)
im_com.paste(
    im_stt_legend, 
    (
        round(im_com.size[0]/2 - im_stt_legend.size[0]/2), 
        round(header_h - 70)
    )
)

im_com.paste(im, (0, header_h))
im_com