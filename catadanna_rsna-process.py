class DataImage():

    def __init__(self):

        self.substudy=''

        self.study=''

        self.file_name=''

        self.pixel_spacing_height=0

        self.pixel_spacing_width=0

        self.slice_thickness=0

        self.np_lung=np.array((512,512))

        self.np_mask=np.array((512,512))

        self.lung_area=0

        self.lung_area_real=0

        self.lung_area_1=0

        self.lung_area_2=0

        self.real_lung_area_1=0

        self.real_lung_area_2=0

        

    def to_string(self):

        print("Class DataImage:")

        print("File name: ", self.file_name) 

        print("Pixel spacing height: ", self.pixel_spacing_height)

        print("Pixel spacing width: ", self.pixel_spacing_width)

        print("Slicethickness: ", self.slice_thickness)

        

        

class DataStudy():

    def __init__(self, s, sbs, w):

        self.study=s

        self.substudy=sbs

        self.where=w # TRAIN or TEST

        self.data_images = [] # list of DataImage

        

    def to_string(self):

        print("Class DataStudy:")

        print("Study: ", self.study)

        print("Subtudy: ", self.substudy)

        print("Where: ", self.where)

        print("Images: ", self.data_images)
def make_model(nh, params):

    hiddens_size = params["hiddens_size"]

    dropout = params["dropout"]

    hiddens_nb = params["hiddens_nb"]

    

    z = L.Input(shape=(nh,), name="Id")

    

    x = L.Dense(hiddens_size, activation="relu", name="d1")(z)

    x = L.BatchNormalization()(x)

    

    if dropout > 0:

        x = L.Dropout(dropout)(x)

    

    if hiddens_nb > 1:

        for i in range(2,hiddens_nb+1):

            x = L.Dense(hiddens_size, activation="relu", name="d"+str(i))(x)

            x = L.BatchNormalization()(x)

            

            if dropout > 0:

                x = L.Dropout(dropout)(x)

    

    x = L.Dense(OUTPUT_SHAPE, activation="sigmoid", name="p1")(x)

    

    model = M.Model(z, x, name="MOA")

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer="adam", metrics=['binary_crossentropy', 'mse'])

    

    return model