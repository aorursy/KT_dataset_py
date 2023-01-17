from fastai.tabular import *

from fastai.callbacks import ReduceLROnPlateauCallback,EarlyStoppingCallback, SaveModelCallback

from sklearn.metrics import roc_auc_score

import joblib

import gc
class roc(Callback):

    

    def on_epoch_begin(self, **kwargs):

        self.total = 0

        self.batch_count = 0

    

    def on_batch_end(self, last_output, last_target, **kwargs):

        preds = F.softmax(last_output, dim=1)

        roc_score = roc_auc_score(to_np(last_target), to_np(preds[:,1]))

        self.total += roc_score

        self.batch_count += 1

    

    def on_epoch_end(self, num_batch, **kwargs):

        self.metric = self.total/self.batch_count
dep_var = 'HasDetections'



cat_names = [ 'RtpStateBitfield','IsSxsPassiveMode','DefaultBrowsersIdentifier',

        'AVProductStatesIdentifier','AVProductsInstalled', 'AVProductsEnabled',

        'CountryIdentifier', 'CityIdentifier', 

        'GeoNameIdentifier', 'LocaleEnglishNameIdentifier',

        'Processor', 'OsBuild', 'OsSuite',

        'SmartScreen','Census_MDC2FormFactor',

        'Census_OEMNameIdentifier', 

        'Census_ProcessorCoreCount',

        'Census_ProcessorModelIdentifier', 

        'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',

        'Census_HasOpticalDiskDrive',

        'Census_TotalPhysicalRAM', 'Census_ChassisTypeName',

        'Census_InternalPrimaryDiagonalDisplaySizeInInches',

        'Census_InternalPrimaryDisplayResolutionHorizontal',

        'Census_InternalPrimaryDisplayResolutionVertical',

        'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',

        'Census_InternalBatteryNumberOfCharges',

        'Census_OSEdition', 'Census_OSInstallLanguageIdentifier',

        'Census_GenuineStateName','Census_ActivationChannel',

        'Census_FirmwareManufacturerIdentifier',

        'Census_IsTouchEnabled', 'Census_IsPenCapable',

        'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer',

        'Wdft_RegionIdentifier', 

        'DateASYear', 'DateASMonth', 'DateASDay', 'DateASIs_year_end',

        'EngineVersion','AppVersion','Census_OSVersion']





procs = [FillMissing, Categorify, Normalize]
data = joblib.load('../input/malwaremodel/data.p')
def bn_drop_lin(n_in:int, n_out:int, bn:bool=False, p:float=0., actn:Optional[nn.Module]=None):

    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."

    layers = [nn.BatchNorm1d(n_in)] if bn else []

    if p != 0: layers.append(nn.AlphaDropout(p))

    layers.append(nn.Linear(n_in, n_out))

    if actn is not None: layers.append(actn)

    return layers
class TabularModel(nn.Module):

    "Basic model for tabular data."

    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,

                  emb_drop:float=0., y_range:OptRange=None, use_bn:bool=False, bn_final:bool=False):

        super().__init__()

        ps = ifnone(ps, [0]*len(layers))

        ps = listify(ps, layers)

        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])

        self.emb_drop = nn.AlphaDropout(emb_drop)

        #self.bn_cont = nn.BatchNorm1d(n_cont)

        n_emb = sum(e.embedding_dim for e in self.embeds)

        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range

        sizes = self.get_sizes(layers, out_sz)

        actns = [nn.SELU(inplace=True)] * (len(sizes)-2) + [None]

        layers = []

        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):

            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)

        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))

        self.layers = nn.Sequential(*layers)

        

    def get_sizes(self, layers, out_sz):

        return [self.n_emb + self.n_cont] + layers + [out_sz]

    

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:

        if self.n_emb != 0:

            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]

            x = torch.cat(x, 1)

            x = self.emb_drop(x)

        if self.n_cont != 0:

            #x_cont = self.bn_cont(x_cont)

            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont

        x = self.layers(x)

        if self.y_range is not None:

            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]

        return x
def tabular_learner(data:DataBunch, layers:Collection[int], emb_szs:Dict[str,int]=None, metrics=None,

         ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=False, **learn_kwargs):

        "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."

        emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))

        model = TabularModel(emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, ps=ps, emb_drop=emb_drop,

                          y_range=y_range, use_bn=use_bn)

        return Learner(data, model, metrics=metrics, **learn_kwargs)
# wd=1e-1

learn = tabular_learner(data, layers=[100, 100, 100, 100, 100, 100, 100, 100], ps=[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05], emb_drop=0.04, metrics=[accuracy], y_range=[0,1.1], model_dir='./', wd=1e-1).to_fp16()
ES = EarlyStoppingCallback(learn, monitor='accuracy',patience = 4)

RLR = ReduceLROnPlateauCallback(learn, monitor='accuracy',patience = 2)

SAVEML = SaveModelCallback(learn, every='improvement', monitor='accuracy', name='best')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(25, 1e-3, callbacks = [ES,RLR, SAVEML])
learn.recorder.plot_losses()
learn.load('best')
preds, _ = learn.get_preds(DatasetType.Test)
sample_submission = pd.read_csv("../input/microsoft-malware-prediction/sample_submission.csv")

#sample_submission.head()
sample_submission.HasDetections = F.softmax(preds, dim=1)[:, 1].numpy()
sample_submission.to_csv('submission-fastai.csv', index=False)