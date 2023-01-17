def denormalize_input(x):

    """

    Converts image pixels from -1:1 range to 0:255 range.

    """

    x /= 2.

    x += 0.5

    x *= 255.

    return x

def load_image_filenames(folders):

    image_files = []

    

    for folder_tuple in folders:

        folder = folder_tuple[0]

        label_index = folder_tuple[1] - 1



        image_filepaths = [os.path.join(folder, image_filename) for image_filename in os.listdir(folder)]

        image_files.extend([(image_filepath, label_index) for image_filepath in image_filepaths])

        

    return image_files



def load_image(filename, target_size):

    try:

        img = image.load_img(filename, target_size=target_size)

    except IOError as e:

        print('Could not read:', filename, ':', e, ', skipping.')

        return None



    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    return x

    

def load_images(image_files, target_size):

    x_list = []

    y_list = []

    

    for image_file in image_files:

        image_filepath = image_file[0]

        label_index = image_file[1]



        x = load_image(image_filepath, target_size)

        if x == None:

            continue

            

        x_list.append(x)



        y = np.zeros((1, 3))

        y[0, label_index] = 1

        y_list.append(y)

        

    X = np.vstack(x_list)

    y = np.vstack(y_list)

    

    return X, y
def denormalize_input(x):

    """

    Converts image pixels from -1:1 range to 0:255 range.

    """

    x /= 2.

    x += 0.5

    x *= 255.

    return x
def show_array(a, fmt='png'):

    """

    Displays an image inside of Jupyter notebook.

    """

    a = np.uint8(a)

    f = StringIO()

    PIL.Image.fromarray(a).save(f, fmt)

    IPython.display.display(IPython.display.Image(data=f.getvalue()))
def show_report(model, X, y):

    """

    Displays a confusion matrix and a classification report.

    """

    y_predicted = np.argmax(model.predict(X), axis=1)

    y_true = np.argmax(y, axis=1)



    print("Confusion matrix (rows: true, columns: predicted)")

    print(confusion_matrix(y_true, y_predicted))

    print("")



    print("Classification report")

    print(classification_report(y_true, y_predicted))
train_folders = [('train/Type_1/', 1), ('train/Type_2/', 2), ('train/Type_3/', 3), ]

               # ('additional/Type_1/', 1), ('additional/Type_2/', 2), ('additional/Type_3/', 3)]
target_size=(224, 224)

train_fraction = 0.8