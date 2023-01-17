!cp ../input/gdcm-conda-install/gdcm.tar .

!tar -xvzf gdcm.tar

!conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2
# import gdcm
from fastai.vision.all import *

from fastai.medical.imaging import *
pd.options.display.max_columns = 100
datapath = Path("/kaggle/input/rsna-str-pulmonary-embolism-detection/")

test_df = pd.read_csv(datapath/'test.csv')

sub_df = pd.read_csv(datapath/'sample_submission.csv')
# train_dcm_files = get_dicom_files(datapath/'train')
# pe_window = (700, 100)

# train_metadata = pd.DataFrame.from_dicoms(train_dcm_files, window=pe_window)
def get_dls(files, size=256, bs=128):

    tfms = [[PILImage.create, ToTensor, RandomResizedCrop(size, min_scale=0.9)], 

            [lambda o: np.random.choice([0,1]), Categorize()]]



    dsets = Datasets(files, tfms=tfms, splits=RandomSplitter(0.1)(files))



    batch_tfms = [IntToFloatTensor]

    dls = dsets.dataloaders(bs=bs, after_batch=batch_tfms)

    return dls
imagepath = Path("/kaggle/input/rsna-str-pe-detection-jpeg-256/")

files = get_image_files((imagepath/'train-jpegs').ls()[0])

files = files[:100]
dls = get_dls(files, bs=64)

learn = cnn_learner(dls, xresnet34, pretrained=False)

learn.path = Path("/kaggle/input/rsnastrpecnnmodel/")

learn.load('basemodel-ft')

learn.to_fp16();
# RGB windows

lung_window = (1500, -600)

pe_window = (700, 100)

mediastinal_window = (400, 40)

windows = (lung_window, pe_window, mediastinal_window)
test_dcm_files = get_dicom_files(datapath/'test')
def get_testdl(test_files, size=256, method='crop', bs=128):

    "At inference time we directly read dcm files not jpg images, so we need a diff get dls func"

    tfms = [[Path.dcmread, partial(DcmDataset.to_nchan, wins=windows, bins=0), Resize(size, method=method)],

            [lambda o: 0, Categorize()]]

    dsets = Datasets(test_files, tfms=tfms, splits=RandomSplitter(0.1)(test_files))

    batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    dls = dsets.dataloaders(bs=bs, after_batch=batch_tfms)

    return dls.test_dl(test_files)
def return_valid_files(dcm_files):

    valid_files = []

    for o in progress_bar(dcm_files):

        try:

            o.dcmread().pixel_array

            valid_files.append(o)

        except:

            pass

    return valid_files
do_full = True

n = 100

submit_full = True



if Path('../input/rsna-str-pulmonary-embolism-detection/train').exists() and not do_full:

    test_dl = get_testdl(return_valid_files(test_dcm_files[:n]), size=256, method='squish', bs=32)

else:

    if submit_full:

        test_dl = get_testdl(return_valid_files(test_dcm_files), size=256, method='squish', bs=32)

    else:

        test_dl = get_testdl(return_valid_files(test_dcm_files[:n]), size=256, method='squish', bs=32)
preds, targs = learn.get_preds(dl=test_dl)
preds[:,1].min(), preds[:,1].max()
study_ids = [o.parent.parent.stem for o in test_dl.items]

instance_ids = [o.stem for o in test_dl.items]

preds_df = pd.DataFrame({"StudyInstanceUID":study_ids, "SOPInstanceUID":instance_ids})

preds_df['pe_present_on_image'] = torch.clamp(preds[:,1], 0.001, 0.999).numpy().astype(np.float64)
preds_df['pe_present_on_image'].min(), preds_df['pe_present_on_image'].max()
preds_df.head()
assert not preds_df.isna().sum().any()
mean_pe = 0.2799

mean_labels = {

             'negative_exam_for_pe': 0.6763928618101033,

             'rv_lv_ratio_gte_1': 0.12875001256566257,

             'rv_lv_ratio_lt_1': 0.17437230326919448,

             'leftsided_pe': 0.21089872969528548,

             'chronic_pe': 0.040139752506710064,

             'rightsided_pe': 0.2575653665766779,

             'acute_and_chronic_pe': 0.019458347341720122,

             'central_pe': 0.054468517151291695,

             'indeterminate': 0.020484822355039723}
study_max_pe = (preds_df.groupby("StudyInstanceUID")['pe_present_on_image'].agg(["max"]).reset_index())
study_max_pe.head()
# max image pe proba for predicted exams

study_max_dict = dict(zip(study_max_pe['StudyInstanceUID'], study_max_pe['max']))
# extract all exam and image ids to predict for

test_unique_sids = test_df['StudyInstanceUID'].unique()

test_unique_sopids = defaultdict(list)

for _,row in test_df.iterrows():

    sid = row['StudyInstanceUID']

    sopid = row['SOPInstanceUID']

    test_unique_sopids[sid].append(sopid)
# prediction dict for each exam - assuming SOPInstanceUID is unique for a given dcm file across all data

pe_image_preds_dict = dict(zip(preds_df['SOPInstanceUID'], preds_df['pe_present_on_image']))
res = []

for sid in test_unique_sids:

    

    sopids = test_unique_sopids[sid]

    

    if sid not in study_max_dict:

        for l in mean_labels: res.append((sid+"_"+l, mean_labels[l]))

        for sopid in sopids: res.append((sopid, mean_pe))

        

    else:

        max_pe = study_max_dict[sid]

        

        if max_pe > 0.5:

            res.append((f"{sid}_negative_exam_for_pe", 1 - mean_labels['negative_exam_for_pe'])) # <=0.5

            res.append((f"{sid}_indeterminate", mean_labels['indeterminate'])) # <=0.5

            

            res.append((f"{sid}_leftsided_pe", 1 - mean_labels['leftsided_pe'])) # >0.5

            res.append((f"{sid}_central_pe", 1 - mean_labels['central_pe'])) # and/or >0.5

            res.append((f"{sid}_rightsided_pe", 1 - mean_labels['rightsided_pe'])) # and/or >0.5

            

            res.append((f"{sid}_rv_lv_ratio_gte_1", 1 - mean_labels['rv_lv_ratio_gte_1'])) # >0.5

            res.append((f"{sid}_rv_lv_ratio_lt_1", mean_labels['rv_lv_ratio_lt_1'])) # or >0.5

            

            res.append((f"{sid}_chronic_pe", mean_labels['chronic_pe'])) # <=0.5 if other >0.5

            res.append((f"{sid}_acute_and_chronic_pe", mean_labels['acute_and_chronic_pe'])) # <=0.5 if other >0.5

            

            for sopid in sopids:

                if sopid in pe_image_preds_dict: res.append((sopid, pe_image_preds_dict[sopid]))

                else:                            res.append((sopid, mean_pe))

            

            

        else:

            res.append((f"{sid}_negative_exam_for_pe", mean_labels['negative_exam_for_pe'])) # >0.5

            res.append((f"{sid}_indeterminate", mean_labels['indeterminate'])) # or >0.5

            

            res.append((f"{sid}_leftsided_pe", mean_labels['leftsided_pe'])) # <=0.5

            res.append((f"{sid}_central_pe", mean_labels['central_pe'])) # and <=0.5

            res.append((f"{sid}_rightsided_pe", mean_labels['rightsided_pe'])) # and <=0.5

            

            res.append((f"{sid}_rv_lv_ratio_gte_1", mean_labels['rv_lv_ratio_gte_1'])) # <=0.5

            res.append((f"{sid}_rv_lv_ratio_lt_1", mean_labels['rv_lv_ratio_lt_1'])) # or <=0.5

            

            res.append((f"{sid}_chronic_pe", mean_labels['chronic_pe'])) # <=0.5 if other >0.5

            res.append((f"{sid}_acute_and_chronic_pe", mean_labels['acute_and_chronic_pe'])) # <=0.5 if other >0.5

            

            for sopid in sopids:

                if sopid in pe_image_preds_dict: res.append((sopid, pe_image_preds_dict[sopid]))

                else:                            res.append((sopid, mean_pe))

        

        

        
new_sub_df = pd.DataFrame(res, columns=['id', 'label'])
assert len(set(sub_df.index).intersection(set(new_sub_df.index))) == len(sub_df)
new_sub_df.head()
def check_consistency(sub, test):

    

    '''

    Checks label consistency and returns the errors

    

    Args:

    sub   = submission dataframe (pandas)

    test  = test.csv dataframe (pandas)

    '''

    

    # EXAM LEVEL

    for i in test['StudyInstanceUID'].unique():

        df_tmp = sub.loc[sub.id.str.contains(i, regex = False)].reset_index(drop = True)

        df_tmp['StudyInstanceUID'] = df_tmp['id'].str.split('_').str[0]

        df_tmp['label_type']       = df_tmp['id'].str.split('_').str[1:].apply(lambda x: '_'.join(x))

        del df_tmp['id']

        if i == test['StudyInstanceUID'].unique()[0]:

            df = df_tmp.copy()

        else:

            df = pd.concat([df, df_tmp], axis = 0)

    df_exam = df.pivot(index = 'StudyInstanceUID', columns = 'label_type', values = 'label')

    

    # IMAGE LEVEL

    df_image = sub.loc[sub.id.isin(test.SOPInstanceUID)].reset_index(drop = True)

    df_image = df_image.merge(test, how = 'left', left_on = 'id', right_on = 'SOPInstanceUID')

    df_image.rename(columns = {"label": "pe_present_on_image"}, inplace = True)

    del df_image['id']

    

    # MERGER

    df = df_exam.merge(df_image, how = 'left', on = 'StudyInstanceUID')

    ids    = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']

    labels = [c for c in df.columns if c not in ids]

    df = df[ids + labels]

    

    # SPLIT NEGATIVE AND POSITIVE EXAMS

    df['positive_images_in_exam'] = df['StudyInstanceUID'].map(df.groupby(['StudyInstanceUID']).pe_present_on_image.max())

    df_pos = df.loc[df.positive_images_in_exam >  0.5]

    df_neg = df.loc[df.positive_images_in_exam <= 0.5]

    

    # CHECKING CONSISTENCY OF POSITIVE EXAM LABELS

    rule1a = df_pos.loc[((df_pos.rv_lv_ratio_lt_1  >  0.5)  & 

                         (df_pos.rv_lv_ratio_gte_1 >  0.5)) | 

                        ((df_pos.rv_lv_ratio_lt_1  <= 0.5)  & 

                         (df_pos.rv_lv_ratio_gte_1 <= 0.5))].reset_index(drop = True)

    rule1a['broken_rule'] = '1a'

    rule1b = df_pos.loc[(df_pos.central_pe    <= 0.5) & 

                        (df_pos.rightsided_pe <= 0.5) & 

                        (df_pos.leftsided_pe  <= 0.5)].reset_index(drop = True)

    rule1b['broken_rule'] = '1b'

    rule1c = df_pos.loc[(df_pos.acute_and_chronic_pe > 0.5) & 

                        (df_pos.chronic_pe           > 0.5)].reset_index(drop = True)

    rule1c['broken_rule'] = '1c'



    # CHECKING CONSISTENCY OF NEGATIVE EXAM LABELS

    rule2a = df_neg.loc[((df_neg.indeterminate        >  0.5)  & 

                         (df_neg.negative_exam_for_pe >  0.5)) | 

                        ((df_neg.indeterminate        <= 0.5)  & 

                         (df_neg.negative_exam_for_pe <= 0.5))].reset_index(drop = True)

    rule2a['broken_rule'] = '2a'

    rule2b = df_neg.loc[(df_neg.rv_lv_ratio_lt_1     > 0.5) | 

                        (df_neg.rv_lv_ratio_gte_1    > 0.5) |

                        (df_neg.central_pe           > 0.5) | 

                        (df_neg.rightsided_pe        > 0.5) | 

                        (df_neg.leftsided_pe         > 0.5) |

                        (df_neg.acute_and_chronic_pe > 0.5) | 

                        (df_neg.chronic_pe           > 0.5)].reset_index(drop = True)

    rule2b['broken_rule'] = '2b'

    

    # MERGING INCONSISTENT PREDICTIONS

    errors = pd.concat([rule1a, rule1b, rule1c, rule2a, rule2b], axis = 0)

    

    # OUTPUT

    print('Found', len(errors), 'inconsistent predictions')

    return errors
res = check_consistency(new_sub_df, test_df)

if len(res) == 0:

    new_sub_df.to_csv("submission.csv", index=False)

else:

    raise("not valid submission")