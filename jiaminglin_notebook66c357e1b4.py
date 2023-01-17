# Load Data

def load_image_data(parent, ids, max_dim = 96, center=True):

    X = np.empty((len(ids), max_dim, max_dim, 1))



    for i, idee in enumerate(ids):

        # Turn the image into an array

        x = resize_img(load_img(os.path.join(parent,'images', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)

        x = img_to_array(x)

        # Get the corners of the bounding box for the image

        length = x.shape[0]

        width = x.shape[1]



        if center:

            h1 = int((max_dim - length) / 2)

            h2 = h1 + length

            w1 = int((max_dim - width) / 2)

            w2 = w1 + width

        else:

            h1, w1 = 0, 0

            h2, w2 = (length, width)

        # Insert into image matrix

        X[i, h1:h2, w1:w2, 0:1] = x

    # Scale the array values so they are between 0 and 1

    return np.around(X / 255.0)



def resize_img(img, max_dim):

    # Get the axis with the larger dimension

    max_ax = max((0, 1), key=lambda i: img.size[i])

    # Scale both axes so the image's largest dimension is max_dim

    scale = max_dim / float(img.size[max_ax])

    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))



def load_table_training(path, standardize = True):

    # Read data from the CSV file

    #zf = ZipFile(path)

    data = pd.read_csv(path)

    ID = data.pop('id')

    # Since the labels are textual, so we encode them categorically

    y = data.pop('species')

    y = LabelEncoder().fit(y).transform(y)

    # standardize the data by setting the mean to 0 and std to 1

    X = StandardScaler().fit(data).transform(data) if standardize else data.values



    return ID, X, y



def load_table_testing(path, standardize = True):

    #zf = ZipFile(path)

    test = pd.read_csv(path)

    ID = test.pop('id')

    # standardize the data by setting the mean to 0 and std to 1

    test = StandardScaler().fit(test).transform(test) if standardize else test.values

    return ID, test



def load_training(parent):

    # Load the pre-extracted features

    file_name = 'train.csv'

    ID, X_table_train, y = load_table_training(os.path.join(parent, file_name))

    # Load the image data

    X_img_train = load_image_data(parent,ID)



    return ID, X_table_train, X_img_train, y



def load_testing(parent):

    # Load the pre-extracted features

    file_name = 'test.csv'

    ID, X_table_test = load_table_testing(os.path.join(parent, file_name))

    # Load the image data

    X_image_test = load_image_data(parent,ID)

    return ID, X_table_test, X_image_test
folder = '.'



epochs = 3

epoch_size = 1000

batch_size = 32



filter_num = 16

filter_dim = 3

activation_func = 'relu'

fc_count = 1

fc_width = 100

dropout_rate = .2

pattern = ['CONV', 'CONV', 'POOL', 'CONV', 'CONV', 'POOL']