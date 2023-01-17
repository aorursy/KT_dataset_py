from IPython.display import JSON

from pathlib import PosixPath

from tqdm import tqdm_notebook



import PIL.ImageDraw

from fastai.vision import *



OCRDCROP = PosixPath("../input/ocrdcrop/")

INPUT = OCRDCROP / "0bags-crop-sauvola"

IMGS_SCALED = OCRDCROP / "scaled"

MASKS_SCALED = OCRDCROP / "masks_scaled"

SCALE_FACTOR = 4



PROCESS_SCALING = False  # already scaled

PROCESS_MASKING = False  # already masked
INPUT.ls()[:5]
imgfiles = sorted(f.relative_to(INPUT) for f in INPUT.ls() if re.match(r'[^.]+\.png$', str(f.relative_to(INPUT))))

binfiles = sorted(f.relative_to(INPUT) for f in INPUT.ls() if re.match(r'[^.]+\.bin\.png$', str(f.relative_to(INPUT))))

annfiles = sorted(f.relative_to(INPUT) for f in INPUT.ls() if re.match(r'[^.]+\.json$', str(f.relative_to(INPUT))))

assert len(imgfiles) == len(binfiles)

assert len(imgfiles) == len(annfiles)

pd.set_option('max_colwidth', 80)

df = pd.DataFrame({'img': imgfiles, 'bin': binfiles, 'ann': annfiles})

df.head()
df["ann_json"] = df.ann.apply(lambda f: json.load(open(INPUT / f, "r")))

df.head()
ann_json_example = json.load(open(INPUT / annfiles[0], "r"))

ann_json_example
segtypes = set()

for aj in df.ann_json.to_list():

    for region in aj.get("regions", []):

        segtypes.add(region["type"])

segtypes = dict((v, k) for k, v in enumerate(sorted(segtypes), start=1))

segtypes["void"] = 0

segtypes
PathLike = Union[str, PosixPath]



def pathify(p: PathLike) -> PosixPath:

    return PosixPath(p) if not type(p) is PosixPath else p 
def resize_image_(imgpath: PosixPath, orig_folder: PathLike = INPUT, dest_folder: PathLike = IMGS_SCALED, scale = SCALE_FACTOR):

    img = open_image(orig_folder / imgpath)

    img.resize((img.shape[0], img.shape[1] // scale, img.shape[2] // scale))

    img.save(dest_folder / imgpath)

if PROCESS_SCALING:

    for f in tqdm_notebook(imgfiles):

        resize_image_(f)
def regions(imgfile: PathLike) -> List[Dict]:

    imgfile = pathify(imgfile)

    annfile = INPUT / re.sub('\.png$', '.json', imgfile.name)

    ann = json.load(open(annfile, "r"))

    return ann.get("regions", [])



def create_mask_(

    imgfile: PathLike,

    classes: Dict[str, int], 

    img_folder: PathLike = INPUT, 

    mask_folder: PathLike = MASKS_SCALED,

    # border: str = "ArtificialBorder", border_width: int = 5,  # XXX(js): no idea about a good border_width

    downscale: int = SCALE_FACTOR,

):

    imgfile = pathify(imgfile)

    res = PIL.Image.open(img_folder / imgfile.name).size

    img = PIL.Image.new(

        mode='L',  # only one 8bit channel (we'll encode the segmentation classes each as one byte with a different nr for each class)

        size=(res[0] // downscale, res[1] // downscale), 

        color=0

    )

    regs = regions(imgfile)

    for r in regs:

        assert "coords" in r

        assert "type" in r

        coords = [(c[0] // downscale, c[1] // downscale) for c in r["coords"]]

        PIL.ImageDraw.Draw(img).polygon(coords, fill=classes[r["type"]])

        

    # Also draw artificial borders but after the filled polygon to not get hidden by any overlapping stuff

    # for r in regs:

    #    coords = r["coords"]

    #    coords += [coords[0], coords[1]]  # necessary to close the lining (it's not autoclosed like for polygonals)

    #    PIL.ImageDraw.Draw(img).line(coords, fill=classes[border], width=border_width)

    

    assert np.max(list(img.getdata())) <= np.max(list(classes.values()))

    img.save(mask_folder / imgfile.name)
if PROCESS_MASKING:

    for f in tqdm_notebook(imgfiles):

        create_mask_(f, segtypes)
RESNET_SIZE = (224, 224)  # that's what resnet is trained for

print("Normalize all image with resize to", RESNET_SIZE)

def get_y_fn(imgfile: PathLike) -> PosixPath:

    return MASKS_SCALED / imgfile.name



def valid_by_book(imgfile: PosixPath, split_pct: float = 0.2) -> bool:

    """ Returns same result for all pages inside a book (given they are in the same folder)"""

    book_name = imgfile.name.split("OCR")[0]  # anything before OCR in "arent_dichtercharaktere_1885_OCR-D-IMG-CROP2_0002.png" determines the book the page is from

    h = int(hashlib.md5(book_name.encode("utf-8")).hexdigest(), 16)  # little trick to calculate a platform independent hash on the name

    return (h % 1e6) / 1e6 < split_pct



def create_data(

    tfms: List[Transform] = None, 

    bs: int = 8,

    sample_p: float = 1.0, split_pct: float = 0.2, split_by_book: bool = False,

    seed: int = None

) -> ImageDataBunch:

    if not tfms: tfms = []

    data= (SegmentationItemList

        .from_folder(IMGS_SCALED)

        .filter_by_rand(sample_p)

    )

    data = (

        data.split_by_valid_func(valid_by_book)

        if split_by_book

        else data.split_by_rand_pct(valid_pct=split_pct, seed=seed)

    )

    data = (data

        .label_from_func(get_y_fn, classes=list(segtypes.values()))

        .transform(tfms, size=RESNET_SIZE, tfm_y=True)

        .databunch(bs=bs)

        .normalize(imagenet_stats)

    )

    return data



data = create_data(seed=1)

data
data.items
data.x[0]
data.y[0]
data.show_batch()
void_code = 0  # I fill image with zeros and we don't want to train this void information

def acc_page_seg(input, target):

    target = target.squeeze(1)

    mask = target != void_code

    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
tfms = []  # no transformations so far

data = create_data(tfms=tfms, split_by_book=True, seed=42)  # XXX(js): tried several batch sizes and 8 seems to work good

data
learn = unet_learner(data, models.resnet34, metrics=acc_page_seg)
learn.fit_one_cycle(4)
learn.show_results()
learn.fit_one_cycle(20, max_lr=slice(1e-6, 1e-4))
learn.save("unet-epochs24")
learn.show_results()
learn.fit_one_cycle(25, max_lr=slice(1e-6, 1e-4))