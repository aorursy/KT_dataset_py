import json

import ast

import base64

import numpy as np

import random

from PIL import Image, ImageDraw, ImageFont

from io import BytesIO
with open('/kaggle/input/ocr-data/extras/single_example/example.json') as f:

    data = json.load(f)



with open('/kaggle/input/ocr-data/extras/single_example/visible_char_map_colors.json') as f:

    colors = json.load(f)

    # Tuples cannot be stored as json, change color assignments to tuples

    colors = {key: tuple(color) for key, color in colors.items()}
data.keys()
data['image_data'].keys()
def convert_string_to_bytes(string):

    """

    Converts a string representation of bytes back to bytes.



    Parameters

    ----------

    string : str

        A string of a bytes-like object.



    Returns

    ----------

    bytes : bytes

        bytes (i.e 'b\'\\x89PNG\\r\\n ... ' -> '\\x89PNG\\r\\n ...').



    """

    return ast.literal_eval(string)





def unpack_list_of_masks(string_list):

    """

    Unpacks a list of string represented bytes objects and returns a list of bytes objects.



    Parameters

    ----------

    string_list : list

        A list of string representation bytes-like objects



    Returns

    ----------

    mask_bytes: list

        A list of png masks as bytes.

    """

    return [convert_string_to_bytes(string) for string in string_list]
def convert_mask_bytes_to_rgba_color_scheme(mask_bytes, label):

    """

    This function makes the png-masks transparent everywhere except the masked area of interest, enhancing the visualization. PNG masks are not inherently RGBA, this function adds the fourth depth of 'alpha' or transparency.

    

    Parameters

    ----------

    mask_bytes : list

        A png masks as bytes.

    label :

        The png mask's corresponding label. (Used for coloring the mask)

    

    Returns

    ----------

    mask: PIL.Image.Image

        An RGBA Image object representing the mask of interest as RGBA, appropriately colored.

    """

    # Open the mask.

    mask = Image.open(BytesIO(mask_bytes))

    mask = mask.convert("RGBA")

    datas = mask.getdata()

    

    newData = []



    # Iterate through the pixel items of the mask. Masks are inverted when saved (i.e. white is the mask, black is the background). Find black pixels and replace with transparent pixels. Replace the mask with the label color.

    for item in datas:

        if item[0] == 0 and item[1] == 0 and item[2] == 0:

            # Make it transparent.

            newData.append((255, 255, 255, 0))

        else:

            # Assign color.

            newData.append(colors[label])



    mask.putdata(newData)

    return mask
def overlay_masks_on_image(img, rgba_masks):

    """

    Modifies the source image in place to show the colored masks for each character.

    

    Parameters

    ----------

    img : PIL.Image.Image

        The source image.

    rgba_masks : list

        A list of masks appropriately converted to RGBA Image objects.



    Returns

    ----------

    None

    """

    for mask in rgba_masks:

        img.paste(mask, (0,0), mask)



def add_text_border(draw_obj, font, text, xmin, ymin):

    """

    Add a thin black border around the text, helps with visualization. Modifies the draw object in place.

    

    Parameters

    ----------

    draw_obj : PIL.ImageDraw.ImageDraw

        The draw object.

    font : PIL.ImageFont.FreeTypeFont

        The ImageFont to add a border to.

    text : str

        The precise text being outlined, generally the label.

    xmin, ymin: int

        The xmin and ymin for the starting point of the text. (Top-Left)

    

    Returns

    ----------

    None

    """

    # Add a thin border.

    draw_obj.text((xmin-2, ymin), text, font=font, fill="black")

    draw_obj.text((xmin+2, ymin), text, font=font, fill="black")

    draw_obj.text((xmin, ymin-2), text, font=font, fill="black")

    draw_obj.text((xmin, ymin+2), text, font=font, fill="black")



def draw_bounding_boxes_on_image(img, xmins, ymins, xmaxs, ymaxs, labels):

    """

    Draws and labels bounding boxes on source image using ground truth lists of details pertaining to the source image. Modifies the source image in place.

    

    Parameters

    ----------

    img : PIL.Image.Image

        The source image.

    xmins, ymins, xmaxs, ymaxs : list

        A list of the respectful coordinates for the image

    labels : list

        A list of labels for each character to be drawn.



    Returns

    ----------

    None

    """

    draw_obj = ImageDraw.Draw(img)

    font_file = "/kaggle/input/ocr-data/extras/single_example/Roboto-Regular.ttf"

    font = ImageFont.truetype(font_file, 32)

    for xmin, ymin, xmax, ymax, label in zip(xmins, ymins, xmaxs, ymaxs, labels):

        draw_obj.rectangle([xmin, ymin, xmax, ymax], outline=colors[label], width=3)

        text = str(label)

        add_text_border(draw_obj, font, text, xmin, ymin)

        draw_obj.text((xmin, ymin), text, font=font, fill=colors[label])
# Open the test image.

example_img = Image.open("/kaggle/input/ocr-data/extras/single_example/example_img.jpg")



# Gather appropriate items from the data json for the image.

xmins = data["image_data"]["xmins_raw"]

ymins = data["image_data"]["ymins_raw"]

xmaxs = data["image_data"]["xmaxs_raw"]

ymaxs = data["image_data"]["ymaxs_raw"]

labels = data['image_data']['visible_latex_chars']



# Unpack and convert serialized pngs.

masks = unpack_list_of_masks(data["image_data"]["png_masks"])
# Convert masks to correct colors and to RGBA format, can take a few seconds.

rgba_masks = [convert_mask_bytes_to_rgba_color_scheme(mask, label) for mask, label in zip(masks, labels)]
# Overlay masks on the image.

overlay_masks_on_image(example_img, rgba_masks)



# Add bounding boxes.

draw_bounding_boxes_on_image(example_img, xmins, ymins, xmaxs, ymaxs, labels)
# Visualize!

display(Image.open("/kaggle/input/ocr-data/extras/single_example/example_img.jpg"))

display(example_img)