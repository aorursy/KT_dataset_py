import numpy as np

import cv2 

import matplotlib.pyplot as plt

import matplotlib.image as image

import math

import os



plt.rcParams['font.sans-serif']=['SimHei']

plt.rcParams['axes.unicode_minus']=False



os.listdir('../input')
img_url = '../input/001_20190415103003_0_0.bmp'

img = cv2.imread(img_url)
# 图像裁剪

cropped = img[9300:13500,900:7000,:]



image = cropped.copy()



# 图像灰度变换

gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) #灰度处理图像

gray_backup = gray.copy()
def threshold_op(gray, mode):

    if mode == 0:  # 全局二值化

        ret, binary = cv2.threshold(gray, 20, 255,

                                   cv2.THRESH_BINARY)  # 手动阈值

    if mode == 1:  # 局部二值化（自适应二值化）

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

                                      cv2.THRESH_BINARY, 51, 10)

    if mode == 2: # 大津法（自动选取阈值）

        retval, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    return binary
binary = threshold_op(gray, 0)

plt.figure(figsize = (16, 10))

plt.imshow(binary,cmap=plt.cm.gray)

plt.title(u'binary figure')



plt.show()
def find_defects_and_contours(find_img, target_img):

    img_contour = target_img.copy()



    contours, hierarchy = cv2.findContours(find_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # # 获取最长轮廓的长度

    # max_len = max(map(lambda x: len(x), contours))

    # # 获取最长的那个轮廓

    # longest_contour = list(filter(lambda x: len(x) == max_len, contours))



    print('We found the %d defects:' %len(contours))



    defects = []

    edges = []

    largest_area = 0

    largest_contour = None

    for i in range(len(contours)):

        defect = contours[i]

        area = cv2.contourArea(defect)

        # 覆盖超过9个像素点的必须检测出来，因此必须取长和宽最小是9

        if area >= 9  and area <6000:   

            defects.append(defect)

            rect = cv2.minAreaRect(defect)  # 外接矩形

            box = np.int0(cv2.boxPoints(rect))  # 获取顶点

            #cv2.drawContours(img_contour, longest_contour, -1, (0, 255, 0), 8)

            cv2.drawContours(img_contour, [box], -1, (255, 0, 0), 5)  # 绘制矩形

        # 定位3个圆孔    

        elif area >= 6000:            

            edges.append(defect)

            if len(defect) > largest_area:

                largest_area = len(defect)

                largest_contour = defect



    edges.remove(largest_contour)     

    

    # 画出3个圆孔

#     for contour in edges:

#         rect = cv2.minAreaRect(contour)  # 外接矩形

#         box = np.int0(cv2.boxPoints(rect))  # 获取顶点

#         cv2.drawContours(img_contour, [box], -1, (0, 255, 0), 5)  # 绘制矩形

        

    # 画出最大的外界矩形

    #cv2.drawContours(img_contour,largest_contour, -1, (0, 0, 255), 9)

    return img_contour, largest_contour, edges
covex_img, largest_contour, edges = find_defects_and_contours(binary, image)





plt.figure(figsize = (80, 80))

plt.subplot(1, 2, 1) 

plt.imshow(covex_img, cmap='gray')



plt.show()



tar_img = covex_img.copy()
fit_img = largest_contour.copy()



print('The shape of contour is:', fit_img.shape)



left_line = []  # 左直线需要match的边框

right_line = [] # 右直线需要match的边框

bottom_line = [] # 底直线需要match的边框



left_circle = [] # 左弧线需要match的边框

right_circle = [] # 右弧线需要match的边框



for i in range(fit_img.shape[0]):

    xs, ys = fit_img[i, 0]

    #  边框 match左竖线的区域

    if xs > 0 and xs <= 300 and ys > 5 and ys <= 3000:

        left_line.append([xs, ys])

    # 边框 match右竖线区域

    elif xs >= 5800 and ys > 5 and ys <= 3000:

        right_line.append([xs, ys])

    # 边框 match 底线区域

    elif xs >= 1000 and xs < 5000 and ys >= 3500:

        bottom_line.append([xs, ys])

    # 边框 match 左底弧区域

    elif xs <= 1000 and ys > 3000:

        left_circle.append([xs, ys])

    # 边框 match 右底弧区域

    elif xs >= 5000 and ys > 3000:

        right_circle.append([xs, ys])    

    

        

left_line, right_line, bottom_line, left_circle, right_circle = np.array(left_line), np.array(right_line), np.array(bottom_line), np.array(left_circle), np.array(right_circle)



print('The shape of left_line is:', left_line.shape)

print('The shape of right_line is:', right_line.shape)

print('The shape of bottom_line is:', bottom_line.shape)

print('The shape of left_circle is:', left_circle.shape)

print('The shape of right_circle is:', right_circle.shape)    
from sklearn.linear_model import LinearRegression



def find_line_defects(line_cor, mode, threshold):

    

    defects = [] # record the left defect



    # 总是让X有大的变化

    if mode == 'left' or mode == 'right':

        x_train, y_train = line_cor[:,1], line_cor[:,0]

    else:

        x_train, y_train = line_cor[:,0], line_cor[:,1]

        

    x_train, y_train = x_train.reshape(-1, 1), y_train.reshape(-1, 1)



    linear_model = LinearRegression().fit(x_train, y_train)



    y_pred = linear_model.predict(x_train)

    distance = np.abs(y_pred - y_train)

    for i in range(line_cor.shape[0]):

        if distance[i,0] > threshold:

            defects.append(line_cor[i].tolist())

    defects = np.array(defects)

        

    # 如果不sort的话，没办法找连续区间，所以一定要先排序

    if mode == 'left' or mode == 'right':

        defects = defects[defects[:,1].argsort()]

        return defects

    else:

        defects = defects[defects[:,0].argsort()]

        return defects
# 找到矩阵的连续区间

def find_continuity(row, ids, array):

    area = []

    length = 1

    

    area.append(array[ids])

    while (ids + 1) < array.shape[0]:

        if abs(array[ids + 1, row] - array[ids, row]) > 1:

            break

        if array[ids + 1, row] - array[ids, row] == 1:

            length += 1

        area.append(array[ids + 1])

        ids += 1

    

    area = np.array(area)

    return area, ids + 1

 

# 找到矩阵的长度超过某阈值的所有连续区间

def fina_all_areas_continue(array, thres, row):

    i = 0

    total = []

    while i < array.shape[0]:

        area, ids = find_continuity(row, i, array)

        if area.shape[0] >= thres:

            total.append(area)

        i = ids

    return total

defect_img = tar_img.copy()



left_defects = find_line_defects(left_line, 'left', 10)   # 寻找左直线的缺陷

right_defects = find_line_defects(right_line, 'right', 10) # 寻找右直线的缺陷

bottom_defects = find_line_defects(bottom_line, 'bottom', 18) # 寻找底直线的缺陷



total_left = fina_all_areas_continue(left_defects, 3, 1)  # 左直线超过3个点的缺陷

total_right = fina_all_areas_continue(right_defects, 3, 1)  # 左直线超过3个点的缺陷

total_bottom = fina_all_areas_continue(bottom_defects, 3, 0)  # 直线超过3个点的缺陷





print(len(total_bottom))



for area in total_left:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(defect_img,(xs - tmp_val,ys),(xe + tmp_val, ye),(255,255,0),5)

    

for area in total_right:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(defect_img,(xs - tmp_val,ys),(xe + tmp_val, ye),(255,255,0),5)

    

for area in total_bottom:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(defect_img,(xs,ys - tmp_val),(xe, ye + tmp_val),(255,255,0),5)

    



plt.figure(figsize = (80, 80))

plt.subplot(1, 2, 1) 

plt.imshow(defect_img, cmap='gray')



plt.show()

def fit_curve_set(cur_set, threshold):

    defects = []

    x_train, y_train = cur_set[:,0], cur_set[:,1]

    

    x_train, y_train = x_train.ravel(), y_train.ravel()



    f1 = np.polyfit(x_train, y_train, 10)

    p1 = np.poly1d(f1)

        

    y_pred = p1(x_train)

    

    y_train, y_pred = y_train.reshape(-1, 1), y_pred.reshape(-1, 1) # 要重新reshape一下，变成(n, 1)的矩阵

    

    distance = np.abs(y_pred - y_train)

    for i in range(cur_set.shape[0]):

        if distance[i,0] > threshold:

            defects.append(cur_set[i].tolist())

    defects = np.array(defects)

    

    

    defects = defects[defects[:,0].argsort()]

    return defects

cur_img = defect_img.copy()



left_curve_defects = fit_curve_set(left_circle, 20)

right_curve_defects = fit_curve_set(right_circle, 20)



total_left_curve = fina_all_areas_continue(left_curve_defects, 3, 0)  # 左曲线超过3个点的缺陷

total_right_curve = fina_all_areas_continue(right_curve_defects, 3, 0)  # 右曲线超过3个点的缺陷



for area in total_left_curve:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(cur_img,(xs- tmp_val,ys - tmp_val),(xe + tmp_val, ye + tmp_val),(0,0,255),5)



for area in total_right_curve:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(cur_img,(xs- tmp_val,ys - tmp_val),(xe + tmp_val, ye + tmp_val),(0,0,255),5)

    



plt.figure(figsize = (80, 80))

plt.subplot(1, 2, 1) 

plt.imshow(cur_img, cmap='gray')



plt.show()
import math



def distance_to_circle(point, center, radius):

    xp, yp = point

    xc, yc = center

    dist = math.sqrt(pow(xp - xc, 2) + pow(yp - yc, 2))

    dist = abs(dist - radius)

    return dist
ecl_img = image.copy()

largest_hole = None

circle_holes= []

length = 0

for edge in edges:

    if edge.shape[0] > length:

        length = edge.shape[0]

        largest_hole = edge

        

for edge in edges:

    if not edge.shape[0] == length:

        circle_holes.append(edge)

        



print('the shape of largest hole is:', largest_hole.shape)



def find_hole_defects(circle_hole, thres):

    ellipse = cv2.fitEllipse(circle_hole)

    hole_defects = []



    center, (low, high), _ = ellipse  

    radius = (high + low) / 4  # 这里high 和low应该理解为直径





    for i in range(circle_hole.shape[0]):

        a, b =  circle_hole[i, 0], circle_hole[i, 1]

        dist = distance_to_circle((a, b), center, radius)

        if dist > thres:

            hole_defects.append(circle_hole[i].tolist())



    hole_defects = np.array(hole_defects)

    if hole_defects.shape[0] > 1:

        hole_defects = hole_defects[hole_defects[:,0].argsort()]

    return hole_defects



#cv2.ellipse(ecl_img,ellipse,(255,255,0),5)
hole_img = cur_img.copy()

first_hole_defects = find_hole_defects(circle_holes[0].reshape(-1,2), 3)

second_hole_defects = find_hole_defects(circle_holes[1].reshape(-1,2), 3)



total_first_hole = fina_all_areas_continue(first_hole_defects, 3, 0)  # 第二个弧查看超过3个点的缺陷

total_second_hole = fina_all_areas_continue(second_hole_defects, 3, 0)  # 第二个弧查看超过3个点的缺陷



for area in total_first_hole:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(hole_img,(xs- tmp_val,ys - tmp_val),(xe + tmp_val, ye + tmp_val),(0,255,0),5)

    

for area in total_second_hole:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(hole_img,(xs- tmp_val,ys - tmp_val),(xe + tmp_val, ye + tmp_val),(0,255,0),5)



    



plt.figure(figsize = (80, 80))

plt.subplot(1, 2, 1) 

plt.imshow(hole_img, cmap='gray')

plt.show()
largest_img = image.copy()

print(largest_hole.shape)

# cv2.drawContours(largest_img, largest_hole, -1, (0,255,0),5)



# plt.figure(figsize = (80, 80))

# plt.subplot(1, 2, 1) 

# plt.imshow(largest_img, cmap='gray')

# plt.show()
largest_img = image.copy()

lc_left, lc_right, lc_top, lc_down = [], [], [], []



for i in range(largest_hole.shape[0]):

    xs, ys = largest_hole[i, 0]

    #  大孔match顶横线区域

    if xs > 700 and xs <= 1500 and ys < 3500:

        lc_top.append([xs, ys])

        

    # 大孔 match底横线区域

    elif xs > 700 and xs <=1500 and ys > 3500:

        lc_down.append([xs, ys])

        

    # 大孔 match 左弧线区域

    elif xs < 700:

        lc_left.append([xs, ys])

        

    # 大孔 match 右弧线区域

    elif xs > 1500:

        lc_right.append([xs, ys])

    

        

lc_left, lc_right, lc_top, lc_down = np.array(lc_left), np.array(lc_right), np.array(lc_top), np.array(lc_down)



print('The shape of hole left is:', lc_left.shape)

print('The shape of hole right is:', lc_right.shape)

print('The shape of hole top is:', lc_top.shape)

print('The shape of hole down is:', lc_down.shape) 



def get_box(points):

    rect = cv2.minAreaRect(points)

    box = np.int0(cv2.boxPoints(rect)) 

    return box



# 以下为测试代码。查看划分是否正确



# box_l = get_box(lc_left)

# box_r = get_box(lc_right)

# box_t = get_box(lc_top)

# box_d = get_box(lc_down)



# cv2.drawContours(largest_img, [box_l], -1, (0,255,0),5)

# cv2.drawContours(largest_img, [box_r], -1, (0,255,0),5)

# cv2.drawContours(largest_img, [box_t], -1, (255,0,0),5)

# cv2.drawContours(largest_img, [box_d], -1, (255,0,0),5)



# plt.figure(figsize = (80, 80))

# plt.subplot(1, 2, 1) 

# plt.imshow(largest_img, cmap='gray')

# plt.show()
def point_line_distance(point_set, w, b):

    '''The point is a numpy set

        w is a value

        b is a value

    '''

    x, y = point_set[:,0], point_set[:,1]

    x, y = x.reshape(-1,1), y.reshape(-1, 1)

    const = np.ones((x.shape[0],1))

    weight =  math.sqrt(pow(w, 2) + pow(-1, 2))

    dist = (1 / weight) * np.abs(w * x + b * const - y)

    return dist   
def p2l_dist_opencv(point_set):

    [vx,vy,x,y] = cv2.fitLine(point_set, cv2.DIST_L2,0,0.01,0.01)

    w = vy / vx

    b = y - w * x

    w, b = w[0], b[0]

    dist = point_line_distance(point_set, w, b)

    return dist
def find_line_defect_cv2(line_set, thres):

    defect = []

    dist = p2l_dist_opencv(line_set)

    for i in range(dist.shape[0]):

        if dist[i,0] > thres:

            defect.append(line_set[i].tolist())

    defect = np.array(defect)

    if defect.shape[0] > 1:

        defect = defect[defect[:,0].argsort()]

    return defect
big_hole_img = hole_img.copy()



top_line_hole_defect, down_line_hole_defect = find_line_defect_cv2(lc_top, 3), find_line_defect_cv2(lc_down, 5)  # 查找最大孔的上直线和下直线的缺陷。这里我下直线可以稍微设置偏差大一点



total_top_line_hole = fina_all_areas_continue(top_line_hole_defect, 3, 0)  # 上直线查看超过3个点的缺陷

total_down_line_hole = fina_all_areas_continue(down_line_hole_defect, 3, 0)  # 下直线查看超过3个点的缺陷



for area in total_top_line_hole:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(big_hole_img,(xs,ys - tmp_val),(xe, ye +  tmp_val),(0,255,0),5)

    

for area in total_down_line_hole:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(big_hole_img,(xs- tmp_val,ys - tmp_val),(xe + tmp_val, ye + tmp_val),(0,255,0),5)



    



plt.figure(figsize = (80, 80))

plt.subplot(1, 2, 1) 

plt.imshow(big_hole_img, cmap='gray')

plt.show()
final_img = big_hole_img.copy()

left_curve_hole_defects = find_hole_defects(lc_left, 5)

right_curve_hole_defects = find_hole_defects(lc_right, 5)



total_left_cur_hole = fina_all_areas_continue(left_curve_hole_defects, 3, 1)  # 查看超左弧个点的缺陷

total_right_cur_hole = fina_all_areas_continue(right_curve_hole_defects, 3, 1)  # 查看右弧超过3个点的缺陷



for area in total_left_cur_hole:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(final_img,(xs- tmp_val,ys - tmp_val),(xe + tmp_val, ye + tmp_val),(0,255,0),5)

    

for area in total_right_cur_hole:

    xs, ys = area[0,0], area[0,1]

    xe, ye = area[-1, 0], area[-1, 1]    

    tmp_val = 15

    cv2.rectangle(final_img,(xs- tmp_val,ys - tmp_val),(xe + tmp_val, ye + tmp_val),(0,255,0),5)



    



plt.figure(figsize = (80, 80))

plt.subplot(1, 2, 1) 

plt.imshow(final_img, cmap='gray')

plt.show()