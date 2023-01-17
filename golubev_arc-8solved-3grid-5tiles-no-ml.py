import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



from skimage.measure import label, regionprops

from itertools import combinations, product

from collections import defaultdict

import matplotlib.pyplot as plt

from matplotlib import colors



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

test_path = data_path / 'test'



def flattener(pred):

    str_pred = str([row for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred
def get_objects(task):

    xs, ys = [], []

    for obj in task['train']:

        xs.append(np.array(obj['input']))

        ys.append(np.array(obj['output']))

    return xs, ys



def find_sub(matrix, sub, ignore = None):

    positions = []

    mask = sub != ignore

    sub_ = sub[mask]

    for x in range(matrix.shape[0]-sub.shape[0]+1):

        for y in range(matrix.shape[1]-sub.shape[1]+1):

            if np.array_equal(matrix[x:x+sub.shape[0], y:y+sub.shape[1]][mask], sub_):

                positions.append((x,y,x+sub.shape[0],y+sub.shape[1]))

    return positions



def check_subitem(task):

    for x, y in zip(*get_objects(task)):

        positions = find_sub(x, y)

        if len(positions) == 0:

            return False

    return True

            

def check_samesize(task):

    for x,y in zip(*get_objects(task)):

        if x.shape != y.shape:

            return False

    return True



def check_sub_mask(task):

    if check_samesize(task):

        return False

    for x,y in zip(*get_objects(task)):

        colors, counts = np.unique(x, return_counts=True)

        found = 0

        for c, area in zip(colors, counts):

            cxs, cys = np.where(x == c)

            xmin,ymin,xmax,ymax = min(cxs),min(cys),max(cxs)+1,max(cys)+1

            shape = (xmax-xmin, ymax-ymin)

            if shape == y.shape and area == np.prod(y.shape):

                found += 1

        if found != 1:

            return False

    return True



def get_cells(x, cols, rows):

    if cols[0] != 0:

        cols = [-1]+cols

    if rows[0] != 0:

        rows = [-1]+rows

    if cols[-1] != x.shape[0]-1:

        cols.append(x.shape[0])

    if rows[-1] != x.shape[1]-1:

        rows.append(x.shape[1])

    cells = np.full((len(cols)-1, len(rows)-1), np.object)

    for i in range(len(cols)-1):

        for j in range(len(rows)-1):

            cells[i][j] = x[cols[i]+1:cols[i+1], rows[j]+1:rows[j+1]]

    return cells



def get_grid(x):

    cols = defaultdict(list)

    rows = defaultdict(list)

    if x.shape[0] < 3 or x.shape[1] < 3:

        return -1, [], []

    for i in range(x.shape[0]):

        if len(np.unique(x[i])) == 1:

            cols[x[i,0]].append(i)

    for i in range(x.shape[1]):

        if len(np.unique(x[:,i])) == 1:

            rows[x[0,i]].append(i)

    for c in cols:

        if c in rows and all(np.diff(cols[c])>1) and all(np.diff(rows[c])>1):

            return c, cols[c], rows[c]

    return -1, [], []



def check_grid(task):

    for x,y in zip(*get_objects(task)):

        color_of_grid, cols, rows = get_grid(x)

        if color_of_grid == -1:

            return False

    return True
def mul_ratio(x, x_ratio):

    x_shape = (x.shape[0]*x_ratio[0], x.shape[1]*x_ratio[1])

    x_ = np.array([x[i//x_ratio[0]][j//x_ratio[1]] for i, j in np.ndindex(x_shape)]).reshape(x_shape)

    return x_



def predict_transforms(xs, ys, test):

    fn = get_transform(xs, ys)

    if fn:

        return [fn(test)]

    ratio = get_ratio(xs, ys)

    if ratio:

        x_ratio, y_ratio = ratio

        xs_ = []

        ys_ = []

        for x, y in zip(xs, ys):

            x, y = mul_ratios(x, y, x_ratio, y_ratio)

            xs_.append(x)

            ys_.append(y)

        fn = get_transform(xs_, ys_)

        if fn:

            test = mul_ratio(test, x_ratio)

            return [fn(test)]

        fns = np.full(x_ratio, np.object)

        for i, j in np.ndindex(x_ratio):

            ys_ = []

            for y in ys:

                m1 = y.shape[0]//x_ratio[0]

                m2 = y.shape[1]//x_ratio[1]

                ys_.append(y[i*m1:(i+1)*m1,j*m2:(j+1)*m2])

            fn = get_transform(xs, ys_)

            if fn:

                fns[i,j] = fn

            else:

                return []

        return [np.concatenate([np.concatenate([fns[i,j](test) for i in range(x_ratio[0])], axis=0) for j in range(x_ratio[1])], axis=1)]

    return []



def predict_grid_transforms(task, test):

    xs, ys = get_objects(task)

    xs = [grid_filter(x) for x in xs]

    return predict_transforms(xs, ys, grid_filter(test))



def get_transform(xs, ys):

    for tranform in get_all_transforms():

        tranformed = True

        for x, y in zip(xs, ys):

            if tranform(x).shape != y.shape:

                tranformed = False

                break

            if not np.equal(tranform(x), y).all():

                tranformed = False

        if tranformed:

            return tranform

    return None



def get_transforms(xs, ys):

    fn = get_transform(xs, ys)

    if fn:

        return fn

    ratio = get_ratio(xs, ys)

    if ratio:

        x_ratio, y_ratio = ratio

        xs_ = []

        ys_ = []

        for x, y in zip(xs, ys):

            x, y = mul_ratios(x, y, x_ratio, y_ratio)

            xs_.append(x)

            ys_.append(y)

        fn = get_transform(xs_, ys_)

        if fn:

            return fn

        fns = np.full(x_ratio, np.object)

        for i, j in np.ndindex(x_ratio):

            ys_ = []

            for y in ys:

                m1 = y.shape[0]//x_ratio[0]

                m2 = y.shape[1]//x_ratio[1]

                ys_.append(y[i*m1:(i+1)*m1,j*m2:(j+1)*m2])

            fn = get_transform(xs, ys_)

            if fn:

                fns[i,j] = fn

            else:

                return None

        return fns

    return None



def check_grid_transforms(task):

    xs, ys = get_objects(task)

    xs = [grid_filter(x) for x in xs]

    return get_transforms(xs, ys) is not None



def get_mode_color(ar):

    colors, counts = np.unique(ar, return_counts=True)

    return colors[np.argmax(counts)]



def grid_filter(x):

    color_of_grid, cols, rows = get_grid(x)

    if color_of_grid == -1:

        return x

    cells = get_cells(x, cols, rows)

    return np.array([get_mode_color(cell) for cell in cells.reshape(-1)]).reshape(cells.shape)



def mul_ratios(x, y, x_ratio, y_ratio):

    x_shape = (x.shape[0]*x_ratio[0], x.shape[1]*x_ratio[1])

    x_ = np.array([x[i//x_ratio[0]][j//x_ratio[1]] for i, j in np.ndindex(x_shape)]).reshape(x_shape)

    y_shape = (y.shape[0]*y_ratio[0], y.shape[1]*y_ratio[1])

    y_ = np.array([y[i//y_ratio[0]][j//y_ratio[1]] for i, j in np.ndindex(y_shape)]).reshape(y_shape)

    return x_, y_



def get_ratio(xs, ys):

    x_ratio = []

    y_ratio = []

    for i in range(2):

        if   all(x.shape[i]%y.shape[i] == 0  for x, y in zip(xs, ys)):

            if len(set(x.shape[i]//y.shape[i] for x, y in zip(xs, ys))) == 1:

                x_ratio.append(1)

                y_ratio.append(xs[0].shape[i]//ys[0].shape[i])

        elif all(y.shape[i]%x.shape[i] == 0  for x, y in zip(xs, ys)):

            if len(set(y.shape[i]//x.shape[i] for x, y in zip(xs, ys))) == 1:

                x_ratio.append(ys[0].shape[i]//xs[0].shape[i])

                y_ratio.append(1)

    if len(x_ratio) != 2:

        return None

    return tuple(x_ratio), tuple(y_ratio)



def check_sub_grid_2x(task):

    if check_samesize(task) or check_subitem(task):

        return False

    for x,y in zip(*get_objects(task)):

        color_of_grid, cols, rows = get_grid(x)

        if color_of_grid == -1:

            return False

        cells = grid_filter(x)

        if (cells.shape[0]*2 != y.shape[0] or cells.shape[1]*2 != y.shape[1]):

            return False

    return True



def check_chess(task, input=False, output=True):

    xs, ys = get_objects(task)

    if input:

        for x in xs:

            if not has_chess(x) and not has_antichess(x):

                return False

    if output:

        for y in ys:

            if not has_chess(y) and not has_antichess(y):

                return False

    return True



def has_chess(g):

    colors = np.unique(g)

    counts = len(colors)

    if counts < 2:

        return False

    indexes = np.zeros(counts, bool)

    for c in colors:

        pts = np.where(g == c)

        s = set([(x+y)%counts for x, y in zip(*pts)])

        if len(s) > 1:

            return False

        index = s.pop()

        if indexes[index]:

            return False

        indexes[index] = True

    return True



def has_antichess(g):

    colors = np.unique(g)

    counts = len(colors)

    if counts < 2:

        return False

    indexes = np.zeros(counts, bool)

    for c in colors:

        pts = np.where(g == c)

        s = set([(g.shape[0]-x+y-1)%counts for x, y in zip(*pts)])

        if len(s) > 1:

            return False

        index = s.pop()

        if indexes[index]:

            return False

        indexes[index] = True

    return True



def find_colors(g):

    colors = np.unique(g)

    counts = len(colors)

    for cnt in range(counts, 1, -1):

        q_colors = np.full(cnt, -1, int)

        for c in colors:

            pts = np.where(g == c)

            s = set([(x+y)%cnt for x, y in zip(*pts)])

            if len(s) > 1:

                continue

            index = s.pop()

            q_colors[index] = c

        

        if -1 not in q_colors:

            return q_colors

    return None



def predict_chess(g):

    q_colors = find_colors(g)

    if q_colors is None:

        colors, counts = np.unique(g, return_counts=True)

        q_colors = colors[np.argsort(counts)][:2]

        

    results = []

    counts = len(q_colors)

    for i in range(counts):

        result = g.copy()

        for x, y in np.ndindex(g.shape):

            result[x,y] = q_colors[(x+y)%counts]

        results.append(result)

        q_colors = np.roll(q_colors, 1)

    return results



def predict_transforms_grid_2x(task, test):

    xs, ys = get_objects(task)

    xs = [grid_filter(x) for x in xs]

    return predict_transforms_2x(xs, ys, grid_filter(test))



def predict_transforms_2x(xs, ys, test):

    predictions = []

    transforms = [

        lambda x: np.rot90(x.T, k=1),

        lambda x: np.rot90(x.T, k=3),

        lambda x: np.rot90(x, k=2),

        lambda x: x,

    ]

    quads = [(1,3,2,0),(3,1,0,2),(2,0,1,3)] # 3 full symmetrical shapes

    for f1,f2,f3,f4 in quads:

        fns = np.array([[transforms[f1],transforms[f2]],[transforms[f3],transforms[f4]]])

        x_ = np.concatenate([np.concatenate([fns[i,j](test) for i in range(2)], axis=0) for j in range(2)], axis=1)

        predictions.append(x_)

    return predictions



def has_repeating(g, ignore=0):

    size0b = int(.6 * g.shape[0])

    size1b = int(.6 * g.shape[1])

    t = np.full((g.shape[0]+2*size0b,g.shape[1]+2*size1b), -1)

    t[size0b:-size0b,size1b:-size1b] = g

    t[t==ignore] = -1

    for size0 in range(2, size0b+1):

        for size1 in range(2, size1b+1):

            for shift0 in range(size0):

                for shift1 in range(size1):

                    pattern = t[size0b+shift0:size0b+shift0+size0,size1b+shift1:size1b+shift1+size1].copy()

                    found = True

                    for d0 in range(size0b+shift0-size0, t.shape[0]-size0, size0):

                        for d1 in range(size1b+shift1-size1, t.shape[1]-size1, size1):

                            test = t[d0:d0+size0,d1:d1+size1]

                            mask = (test != -1) & (pattern != -1)

                            if np.array_equal(test[mask], pattern[mask]):

                                ind = test != -1

                                pattern[ind] = test[ind]

                            else:

                                found = False

                                break

                    if found:

                        return shift0, shift1, pattern

    return None



def check_repeating(task, has_complete=False):

    patterns = []

    for x, y in zip(*get_objects(task)):

        if len(np.unique(x)) < 3 or not has_repeating(y,-1):

            return False

        result = None

        for c in np.unique(x):

#             if c not in np.unique(y):

            result = has_repeating(x,c)

            if result:

                sh0,sh1,pattern = result

                pattern[pattern == -1] = c

                if has_complete:

                    pred = np.tile(pattern, (x.shape[0]//pattern.shape[0]+2, x.shape[1]//pattern.shape[1]+2))

                    pred1 = pred[sh0:sh0+x.shape[0],sh1:sh1+x.shape[1]]

                    pred2 = pred[sh0+1:sh0+1+x.shape[0],sh1:sh1+x.shape[1]]

                    pred3 = pred[sh0:sh0+x.shape[0],sh1+1:sh1+1+x.shape[1]]

                    if np.array_equal(pred1, y) or np.array_equal(pred2, y) or np.array_equal(pred3, y):

                        break

                    result = None

                else:

                    break

        if not result:

            return False

    return True



def predict_repeating(x):

    for c in np.unique(x):

        result = has_repeating(x, c)

        if result:

            sh0,sh1,pattern = result

            pattern[pattern == -1] = c

            pred = np.tile(pattern, (x.shape[0]//pattern.shape[0]+2, x.shape[1]//pattern.shape[1]+2))

            pred1 = pred[sh0:sh0+x.shape[0],sh1:sh1+x.shape[1]]

            pred2 = pred[sh0+1:sh0+1+x.shape[0],sh1:sh1+x.shape[1]]

            pred3 = pred[sh0:sh0+x.shape[0],sh1+1:sh1+1+x.shape[1]]

            return [pred1,pred2,pred3]

    return []



def predict_repeating_mask(x):

    predictions = predict_repeating(x)

    if len(predictions) > 0:

        rows, cols = np.where(predictions[0] != x)

        return [predictions[0][min(rows):max(rows)+1,min(cols):max(cols)+1]]

    return []



def trim_matrix(x):

    if len(np.unique(x)) == 1:

        return x

    for c in np.unique(x):

        xs,ys = np.where(x!=c)

        xmin,ymin,xmax,ymax = min(xs),min(ys),max(xs)+1,max(ys)+1

        if xmin > 0 or ymin > 0 or xmax < x.shape[0] or ymax < x.shape[1]:

            return x[xmin:xmax,ymin:ymax]

    return x



def trim_matrix_box(g, mask=None):

    if mask is None:

        mask = np.unique(g)

    if len(np.unique(g)) == 1:

        return None

    for c in mask:

        xs,ys = np.where(g!=c)

        xmin,ymin,xmax,ymax = min(xs),min(ys),max(xs)+1,max(ys)+1

        if xmin > 0 or ymin > 0 or xmax < g.shape[0] or ymax < g.shape[1]:

            return (xmin,ymin,xmax,ymax)

    return None



def has_tiles(g, ignore=0):

    for size0b, size1b in [(g.shape[0], int(0.6*g.shape[1])), (int(0.6*g.shape[0]), g.shape[1])]:

        t = np.full((g.shape[0]+size0b, g.shape[1]+size1b), -1)

        t[:-size0b,:-size1b] = g

        t[t==ignore] = -1

        box_trim = trim_matrix_box(g,[ignore])

        min_size0 = 1

        min_size1 = 1

        if box_trim is not None and ignore != -1:

            xmin,ymin,xmax,ymax = box_trim

            t[xmin:xmax,ymin:ymax] = g[xmin:xmax,ymin:ymax]

            min_size0 = xmax-xmin

            min_size1 = ymax-ymin

        for size0 in range(min_size0, size0b+1):

            for size1 in range(min_size1, size1b+1):

                pattern = t[:size0,:size1].copy()

                found = True

                for d0 in range(0, t.shape[0]-size0, size0):

                    for d1 in range(0, t.shape[1]-size1, size1):

                        test = t[d0:d0+size0,d1:d1+size1]

                        mask = (test != -1) & (pattern != -1)

                        if np.array_equal(test[mask], pattern[mask]):

                            ind = test != -1

                            pattern[ind] = test[ind]

                        else:

                            found = False

                            break

                if found:

                    return pattern

    return None



def roll_color(g):

    from_values = np.unique(g)

    to_values = np.roll(from_values, 1)



    sort_idx = np.argsort(from_values)

    idx = np.searchsorted(from_values, g, sorter = sort_idx)

    return to_values[sort_idx][idx]



def get_all_transforms():

    return [

        lambda x: roll_color(x),

        lambda x: np.roll(x, -1, axis=0),

        lambda x: np.roll(x,  1, axis=0),

        lambda x: np.roll(x, -1, axis=1),

        lambda x: np.roll(x,  1, axis=1),

        lambda x: np.rot90(x.T, k=1),

        lambda x: np.rot90(x.T, k=2),

        lambda x: np.rot90(x.T, k=3),

        lambda x: np.rot90(x.T, k=4),

        lambda x: np.rot90(x, k=1),

        lambda x: np.rot90(x, k=2),

        lambda x: np.rot90(x, k=3),

        lambda x: x,

    ]



def has_tiles_shape(g, shape, ignore=0):

    for size0b, size1b in [(g.shape[0], int(0.6*g.shape[1])), (int(0.6*g.shape[0]), g.shape[1])]:

        t = np.full((g.shape[0]+size0b, g.shape[1]+size1b), -1)

        t[:-size0b,:-size1b] = g

        t[t==ignore] = -1

        box_trim = trim_matrix_box(g,[ignore])

        min_size0 = 1

        min_size1 = 1

        if box_trim is not None and ignore != -1:

            xmin,ymin,xmax,ymax = box_trim

            t[xmin:xmax,ymin:ymax] = g[xmin:xmax,ymin:ymax]

            min_size0 = xmax-xmin

            min_size1 = ymax-ymin

        size0 = shape[0]

        size1 = shape[1]

        pattern = t[:size0,:size1].copy()

        found = True

        for d0 in range(0, t.shape[0]-size0, size0):

            for d1 in range(0, t.shape[1]-size1, size1):

                test = t[d0:d0+size0,d1:d1+size1]

                mask = (test != -1) & (pattern != -1)

                if np.array_equal(test[mask], pattern[mask]):

                    ind = test != -1

                    pattern[ind] = test[ind]

                else:

                    found = False

                    break

        if found:

            return pattern

    return None



def check_tiles_shape(task, has_complete=0): 

    patterns = []

    for x, y in zip(*get_objects(task)):

        o_pattern = has_tiles(y,-1)

        if len(np.unique(x)) < 2 or o_pattern is None:

            return False

        found = False

        for c in [-1, *np.unique(x)]:

            pattern = has_tiles_shape(x, o_pattern.shape, c)

            if pattern is not None:

                pattern[pattern == -1] = c

                if has_complete:

                    for transform in get_all_transforms():

                        transformed_pattern = transform(pattern)

                        pred = np.tile(transformed_pattern, (x.shape[0]//transformed_pattern.shape[0]+2, x.shape[1]//transformed_pattern.shape[1]+2))

                        pred = pred[:x.shape[0],:x.shape[1]]

                        if np.array_equal(pred, y):

                            found = True

                            patterns.append(pattern)

                            break

                else:

                    found = True

                    patterns.append(pattern)

        if not found:

            return False

    return True



def predict_tiles_shape(task, test_input):

    has_transforms = set()

    has_shapes = set()

    for x, y in zip(*get_objects(task)):

        o_pattern = has_tiles(y,-1)

        if len(np.unique(x)) < 2 or o_pattern is None:

            return []

        found = False

        for c in [-1, *np.unique(x)]:

            pattern = has_tiles_shape(x, o_pattern.shape, c)

            if pattern is not None:

                pattern[pattern == -1] = c

                for transform in get_all_transforms():

                    transformed_pattern = transform(pattern)

                    pred = np.tile(transformed_pattern, (x.shape[0]//transformed_pattern.shape[0]+2, x.shape[1]//transformed_pattern.shape[1]+2))

                    pred = pred[:x.shape[0],:x.shape[1]]

                    if np.array_equal(pred, y):

                        found = True

                        has_transforms.add(transform)

                        has_shapes.add(o_pattern.shape)

                        break

        if not found:

            return []

     

    preds = []

    for c in np.unique(test_input):

        for shape in has_shapes:

            pattern = has_tiles_shape(test_input, shape, c)

            if pattern is None:

                continue

            pattern[pattern == -1] = c



            pred = np.tile(pattern, (test_input.shape[0]//pattern.shape[0]+2, test_input.shape[1]//pattern.shape[1]+2))

            for transform in has_transforms:

                transformed_pattern = transform(pattern)

                pred = np.tile(transformed_pattern, (test_input.shape[0]//transformed_pattern.shape[0]+2, test_input.shape[1]//transformed_pattern.shape[1]+2))

                pred = pred[:test_input.shape[0],:test_input.shape[1]]

                preds.append(pred)

    return preds
outputs = []

submission = pd.read_csv(data_path/'sample_submission.csv')

problems = submission['output_id'].values

for i in  range(len(problems)):

    output_id = problems[i]

    task_id = output_id.split('_')[0]

    pair_id = int(output_id.split('_')[1])

    f = str(test_path / str(task_id + '.json'))

    with open(f, 'r') as read_file:

        task = json.load(read_file)

        for key_task in task:

            for obj in task[key_task]:

                for key in obj:

                    obj[key] = np.array(obj[key])

    test_input = task['test'][pair_id]['input']

    predictions = []

    if check_sub_mask(task):

        predictions.extend(predict_repeating_mask(test_input))

    elif check_grid(task) and check_grid_transforms(task):

        predictions.extend(predict_grid_transforms(task, test_input))

    elif check_grid(task) and check_chess(task, False, True):

        predictions.extend(predict_chess(grid_filter(test_input)))

    elif check_grid(task) and check_sub_grid_2x(task):

        predictions.extend(predict_transforms_grid_2x(task, test_input))

    elif check_repeating(task, True):

        predictions.extend(predict_repeating(test_input))

    elif check_tiles_shape(task, True):

        predictions.extend(predict_tiles_shape(task, test_input))

    outputs.append(' '.join(list(set(flattener(pred.tolist()) for pred in predictions))[:3]))

submission['output'] = outputs

submission.to_csv('submission.csv', index=False)