import math

import os

import cv2

import csv

import numpy as np

from sklearn.cluster import KMeans

from collections import Counter

from glob import glob
def viewImage(image, title):

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    cv2.imshow(title, image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()





def recizeImg(image):

    width = 300

    height = 300

    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    return resized





def get_dominant_color(image, k=4):

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters=k)

    labels = clt.fit_predict(image)

    label_counts = Counter(labels)

    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)





# circle_val

def circle_measure(cnt):

    denominator = cv2.arcLength(cnt, True) ** 2

    return math.pi * 4 * cv2.contourArea(cnt) / denominator if denominator != 0 else 0





# med area react for results:

def med_area_rectangles(little_rects):

    res_area = 0

    for ltl_rect in little_rects:

        x, y, w, h = ltl_rect

        res_area += w * h

    return res_area / len(little_rects)





def rect_area(rect):

    x, y, w, h = rect

    return w * h





def median_for_contours_shape(cnts):

    widths, height = 0, 0

    for cnt in cnts:

        _, _, w, h = cv2.boundingRect(cnt)

        widths += w

        height += h

    return widths / len(cnts), height / len(cnts)





def resolve_ret(dom_color, lim_yell_color):

    ret = 175 if abs(dom_color[0] - lim_yell_color[0]) > 6 else 140

    return ret





def del_little_area_contours(contours):

    # find med area and del some contours

    res_area = 0

    for cnt in contours:

        res_area += cv2.contourArea(cnt)

    med_area = res_area / len(contours)

    ind = 0

    while ind != len(contours):

        if cv2.contourArea(contours[ind]) < med_area:

            del contours[ind]

        else:

            ind += 1





def take_circle_measure_in_little_cnts(little_cnts, x, y, w, h, ltl_rectangles):

    ind = 0

    while ind != len(little_cnts):

        if circle_measure(little_cnts[ind]) < 0.3:

            del little_cnts[ind]

        else:

            ltl_rect = x_little, y_little, w_little, h_little = cv2.boundingRect(little_cnts[ind])

            if ltl_rect != (x, y, w, h):

                ltl_rectangles.append([x_little + x, y_little + y, w_little, h_little])

            ind += 1





def del_small_rects_in_little_cnts(ltl_rectangles, img_draw_first):

    med_area_rects = med_area_rectangles(ltl_rectangles)

    for rect in ltl_rectangles:

        if rect_area(rect) <= med_area_rects * 1.5:

            x_little, y_little, w_little, h_little = rect

            cv2.rectangle(img_draw_first, (x_little, y_little), (x_little + w_little, y_little + h_little),

                          (0, 255, 0), 2)





def draw_big_rects_orign(big_rectangles, img):

    for big_rect in big_rectangles:

        x, y, w, h = big_rect

        cv2.rectangle(img, (x, y), (x + w, y + h),

                      (0, 255, 0), 2)





def try_work_with_every_corn(contours, target):

    ltl_rectangles = []

    med_w, med_h = median_for_contours_shape(contours)

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        roi_color = target[y:y + h, x:x + w]

        roi_color_copy = roi_color.copy()

        # viewImage(roi_color, title='one contour')

        if w < med_w and h < med_h:  # if one img shape less then med then skip him

            continue

        gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)  # swing effect

        # viewImage(gray, title='gray one contour')

        edged = cv2.Canny(gray, 10, 100, L2gradient=True)  # seek contours in gray img(without mask)

        # viewImage(edged, title='gray one contour with limits')

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)  # for closing open limits

        # viewImage(closed, title='morfology')

        little_cnts = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)[0]

        cv2.drawContours(roi_color, little_cnts, -1, (0, 0, 255), 1)

        # viewImage(roi_color, title='small cnts on one cnt')



        # take only circle forms

        take_circle_measure_in_little_cnts(little_cnts, x, y, w, h, ltl_rectangles)



        big_rectangles.append([x, y, w, h])

        cv2.drawContours(roi_color_copy, little_cnts, -1, (0, 0, 255), 1)

        # viewImage(roi_color_copy, title='small cnts on one cnt who have circle')

    return ltl_rectangles, big_rectangles





def get_big_rectangles(contours):

    big_rectangles = []

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        big_rectangles.append([x, y, w, h])

    return big_rectangles





def read_images(path):

    titles = []

    images = []

    count = 0

    for image_path in glob(os.path.join(path, "*.jpg")):

        titles.append(os.path.basename(image_path)[:-4])

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

        images.append(image_bgr)

        count += 1

    return images, titles
img_num = 1

lim_yell_color = [44.08729689253203, 107.89121749458928, 43.02856063060145]

path = "../input/global-wheat-detection/test"

images, titles = read_images(path)

title_index = 0
title_index = 0

with open('submission.csv', mode='tw', newline='') as employee_file:

    employee_writer = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

    employee_writer.writerow(['image_id', 'PredictionString'])



    for img in images:

        img_draw_first = img.copy()

        img_draw_second = img.copy()



        # resize image, maybe use in future

        # img = recizeImg(img)



        # find dom_color

        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # dom_color = get_dominant_color(hsv_image)

        dom_color = lim_yell_color





        # print orig and hsv

        # viewImage(img, title='original(recize)')

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # viewImage(hsv_img, title='hsv_img')



        # check yell_hsv

        yell = np.uint8([[[0, 237, 255]]])

        yell_hsv = cv2.cvtColor(yell, cv2.COLOR_BGR2HSV)

        # print('yell_hsv : ', yell_hsv)



        ## getting green HSV color representation

        yell_low = np.array([20, 0, 100])

        yell_high = np.array([40, 255, 255])

        curr_mask = cv2.inRange(hsv_img, yell_low, yell_high)

        hsv_img[curr_mask > 0] = ([30, 255, 255])

        # Vizualize the mask

        # viewImage(curr_mask, title='mask')

        # viewImage(hsv_img, title='with mask')

        target = cv2.bitwise_and(img, img, mask=curr_mask)

        # cv2.imwrite("target.png", target)  # сегментированное изображение



        ## converting the HSV image to Gray inorder to be able to apply

        ## contouring

        RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

        gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)

        # viewImage(gray, title='to gray')  ## 3

        ret = resolve_ret(dom_color, lim_yell_color)

        ret, threshold = cv2.threshold(gray, ret, 255, 0)  # если изобрыжение слишком темное(после наложенной маски),

        # то оно закрашивается в черный цвет

        # viewImage(threshold, title='limit_picture')  ## пороговое изображение, ret - порог

        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        # find med area and del some contours

        del_little_area_contours(contours)

        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

        # viewImage(img, title='conturs after del little area')

        # cv2.imwrite("just_red_cntrs.png", img)

        # viewImage(target, title='conturs on target')



        # work either cnt:

        big_rectangles = get_big_rectangles(contours)





        # del_small_rects_in_little_cnts(ltl_rectangles, img_draw_first)

        # viewImage(img_draw_first, title='res')

        draw_big_rects_orign(big_rectangles, img_draw_second)

        # viewImage(img_draw_second, title='res with rects')

        # cv2.imwrite("img_with_little_rect.png", img_draw_first)

        # cv2.imwrite(path + 'results/' + titles[title_index] + "_big_rect.png", img_draw_second)



        #save svc

        title_str = str(titles[title_index])

        out = ""

        for ind, rect in enumerate(big_rectangles):

#             if ind > 40:

#                 break

            x, y, w, h = map(lambda x: str(x), rect)

            first_num = str(0.5)

            out += ' '.join([first_num, x, y, w, h, ' '])

        employee_writer.writerow([title_str, out])

        title_index += 1