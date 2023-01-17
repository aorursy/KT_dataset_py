import numpy as np
import pandas as pd
train = pd.read_csv('../input/training.csv')
test = pd.read_csv('../input/test.csv')

trainids = train.id.ravel()
testids = test.id.ravel()
trainsignals = train.signal.ravel()
train.drop(['id','signal'],inplace=True,axis=1)
test.drop('id',inplace=True,axis=1)
def Output(p):
    return 1/(1.+np.exp(-p))

def GP(data):
    return Output(  1.0*np.tanh(((((((((data["IPSig"]) + (data["ISO_SumBDT"]))) - (np.minimum(((-2.0)), ((data["ISO_SumBDT"])))))) / (data["ISO_SumBDT"]))) / (np.minimum((((-1.0*((data["ISO_SumBDT"]))))), ((data["IPSig"])))))) +
                    1.0*np.tanh((-1.0*((((data["iso"]) + (((((((((((((data["VertexChi2"]) + ((3.0)))) / (data["ISO_SumBDT"]))) * (data["IP"]))) * 2.0)) / (data["ISO_SumBDT"]))) * (((((((((((data["VertexChi2"]) + ((3.0)))) / (data["ISO_SumBDT"]))) * (data["IP"]))) * 2.0)) / (data["ISO_SumBDT"])))))))))) +
                    1.0*np.tanh((-1.0*(((((((((data["IPSig"]) * ((((data["iso"]) + (((data["IP"]) * 2.0)))/2.0)))) + (np.tanh((data["p0_IsoBDT"]))))/2.0)) * ((((data["p0_IsoBDT"]) + (data["IPSig"]))/2.0))))))) +
                    1.0*np.tanh(((np.minimum(((np.cos((((np.cos((((data["p0_track_Chi2Dof"]) * (np.cos((data["p0_track_Chi2Dof"]))))))) * (np.log((data["IP_p0p2"])))))))), ((np.cos((data["p0_track_Chi2Dof"])))))) * (data["p0_track_Chi2Dof"]))) +
                    1.0*np.tanh((((((((((11.05855369567871094)) / (((((11.05855751037597656)) + (((((data["SPDhits"]) / 2.0)) / 2.0)))/2.0)))) - (data["IP"]))) - (((data["SPDhits"]) / (data["p1_pt"]))))) * 2.0)) +
                    1.0*np.tanh((((((((((((((data["CDF3"]) / (data["dira"]))) > (data["CDF3"]))*1.)) > (data["CDF3"]))*1.)) / 2.0)) + ((-1.0*((((((data["CDF3"]) * (data["p2_track_Chi2Dof"]))) * (((data["CDF3"]) * (data["p2_track_Chi2Dof"])))))))))/2.0)) +
                    1.0*np.tanh((((-1.0*((((data["DOCAthree"]) / (data["CDF2"])))))) + (np.minimum(((((data["p2_pt"]) / (data["p0_p"])))), ((np.minimum(((data["CDF2"])), ((((np.sin((1.570796))) / 2.0)))))))))) +
                    1.0*np.tanh(np.minimum((((-1.0*(((((((data["FlightDistance"]) < (data["IPSig"]))*1.)) / 2.0)))))), ((((np.minimum(((np.cos((np.log((data["p0_pt"])))))), ((np.cos((data["p1_track_Chi2Dof"])))))) / (0.318310)))))) +
                    1.0*np.tanh(((np.sin((np.where(data["iso"]>0, ((((data["iso"]) - ((-1.0*((((data["IPSig"]) / 2.0))))))) / 2.0), ((((3.0) * (data["IP"]))) * 2.0) )))) / 2.0)) +
                    1.0*np.tanh(((((np.cos(((((data["ISO_SumBDT"]) + (0.318310))/2.0)))) - (np.sin((np.log((data["p1_eta"]))))))) - ((((((data["ISO_SumBDT"]) + (np.cos((data["p2_IsoBDT"]))))/2.0)) * ((((data["ISO_SumBDT"]) + (np.cos((data["p2_IsoBDT"]))))/2.0)))))))
pd.DataFrame({'id':testids,'prediction':GP(test).values}).to_csv('minimal.csv',index=False)