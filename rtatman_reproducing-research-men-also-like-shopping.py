import os

# change directory to the dataset where our
# custom scripts are found
os.chdir("/kaggle/input/utility-scripts/")

# read in custom modules 
import fairCRF_utils as myutils
import inference_debias as cocoutils
import preprocess as mypreprocess

# reset our working directory
os.chdir("/kaggle/working/")
# GPU is only needed for Caffe stuff (which we' can't do)
# myutils.set_GPU(1)
margin = 0.05
vSRL = 1
is_dev = 1
#myutils.show_amplified_bias(margin, vSRL, is_dev)
margin = 0.05
vSRL = 0
is_dev = 1
myutils.show_amplified_bias(margin, vSRL, is_dev)
def lagrange_with_margins(margin, eta, constraints, inference, get_acc, get_update_index, 
                          all_man_idx, all_woman_idx, arg_inner_all, vSRL, *args):
    lambdas = {item:[0,0] for item in constraints}
    arg_inner_tmp = arg_inner_all.copy()
    results = []
    if vSRL == 1:
        inf_arg = (args[0:-1])
        acc_arg = (args[0:2]) + (args[-1],)
    else:
        inf_arg = ""
        acc_arg = (args[0],)
        
    print("Starting Lagrangian part")     
    for epoch in range(100):
        count = 0
        error = {item:[0,0] for item in constraints}

        top1, pred_agents = inference(arg_inner_tmp, *inf_arg) 
        non_zeros = {}
        for k in constraints:
            if k in pred_agents:
                    lambdas[k][0] += eta * constraints[k][0][0] * pred_agents[k][0]
                    lambdas[k][1] += eta * constraints[k][1][0] * pred_agents[k][0]
                    error[k][0] += constraints[k][0][0] * pred_agents[k][0]
                    error[k][1] +=  constraints[k][1][0] * pred_agents[k][0]
                    lambdas[k][0] += eta * constraints[k][0][1] * pred_agents[k][1]
                    lambdas[k][1] += eta * constraints[k][1][1] * pred_agents[k][1]
                    error[k][0] += constraints[k][0][1] *  pred_agents[k][1]
                    error[k][1] +=  constraints[k][1][1] *  pred_agents[k][1]
        
        for k in lambdas:
            for i in range(2):
                if lambdas[k][i] <= 0:
                    lambdas[k][i] = 0

        for k in error:
            for i in range(2):
                if error[k][i] > 0:
                    count += 1

        arg_inner_tmp = arg_inner_all.copy()

        for i in range(len(arg_inner_tmp)):
            for arg_idx in top1[i][1]:
                if arg_idx in all_man_idx:
                    k = get_update_index(top1, i, arg_idx, 1, vSRL)
                    if k in lambdas:  
                        arg_inner_tmp[i][arg_idx] -= lambdas[k][0] * constraints[k][0][0] 
                        arg_inner_tmp[i][arg_idx] -= lambdas[k][1] * constraints[k][1][0]
                if arg_idx in all_woman_idx:
                    k = get_update_index(top1, i, arg_idx, 0, vSRL)
                    if k in lambdas: 
                        arg_inner_tmp[i][arg_idx] -= lambdas[k][0] * constraints[k][0][1] 
                        arg_inner_tmp[i][arg_idx] -= lambdas[k][1] * constraints[k][1][1]

        if epoch % 10 == 0 or epoch == 99:
            print("%s-th epoch, number of times that constrints are not satisfied:"%(epoch), count)
            acc1 = get_acc(arg_inner_tmp, *acc_arg)
            print("%s-epoch, acc is: "%(epoch),acc1)
            results.append([epoch, count, acc1])
            
        if count == 0:
            break
    myutils.save_iterations(myutils.configs['save_iteration'] + "_margin_" + str(margin), results)
    myutils.save_lambdas(myutils.configs['save_lambda'] + "_margin_" + str(margin), lambdas)
    return arg_inner_tmp, lambdas
def run(margin, vSRL, is_dev, eta):
    reargs = mypreprocess.preprocess(margin, vSRL, is_dev)
    if vSRL != 1:
        eta = 0.05
        (constraints, all_man_idx, all_woman_idx, arg_inner_all, 
         target, pred_objs_bef, cons_verbs, train_samples) = reargs
        arg_inner_tmp, lambdas = lagrange_with_margins(margin, eta, constraints, 
                                                       cocoutils.inference, cocoutils.accuracy,
                                                       myutils.get_update_index, all_man_idx, 
                                                       all_woman_idx, arg_inner_all, vSRL, target)
        
        mypreprocess.show_results(margin, vSRL, arg_inner_tmp, cons_verbs, 
                                  train_samples, pred_objs_bef)
    else:
        eta = 0.1
        (arg_inner_all, value_frame_all, label_all, len_verb_file, all_man_idx, all_woman_idx, 
         constraints, output_index, id_verb, verb_roles, cons_verbs, num_gender,words_file, 
         training_file, role_potential_file, verb_id) = reargs
        value_frame_tmp = value_frame_all.copy()
        label = label_all.copy()
        arg_inner_tmp, lambdas = lagrange_with_margins(margin, eta, constraints, 
                                                       myutils.inference, myutils.get_acc, 
                                                       myutils.get_update_index,
                                                       all_man_idx, all_woman_idx, arg_inner_all, 
                                                       vSRL, value_frame_tmp, label,output_index, 
                                                       id_verb, verb_roles, len_verb_file)
        
        mypreprocess.show_results(margin, vSRL, cons_verbs, num_gender, words_file, 
                                  training_file, role_potential_file, arg_inner_all, value_frame_all,
                                  label_all, arg_inner_tmp, value_frame_all, output_index, id_verb, 
                                  verb_id, verb_roles)

margin = 0.05
is_dev = 1
#run(margin, 1, is_dev, 0.1)
margin = 0.05
is_dev = 1
run(margin, 0, is_dev, 0.05)