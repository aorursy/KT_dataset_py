# These modules are used to perform Flask web server

from flask import Flask, jsonify, request,render_template

#this module is used to interact with mysql database

import mysql.connector

#This module contains database access information
#Configuration file containing DB access credentials

username="root"

password="password"

database_name="backend_dev_db_pr"

#Required parameters

req_l_par=(("node_id","language"))

#Optional parameters

opt_l_par=(("page_num","page_size","search_keyword"))
def input_acq():

    input_par=dict()

    #Required paremeters (see req_l_par global tuple)

    input_par[req_l_par[0]]=request.args.get(req_l_par[0])

    input_par[req_l_par[1]]=request.args.get(req_l_par[1])

   #Optional paremeters (see opt_l_par global tuple)

    input_par[opt_l_par[2]]=request.args.get(opt_l_par[2])

    input_par[opt_l_par[0]]=request.args.get(opt_l_par[0])

    input_par[opt_l_par[1]]=request.args.get(opt_l_par[1])

    return input_par 
def set_input(in_p_dict):

    

    for i in in_p_dict:

        if(in_p_dict[i] is None):

            if(i in req_l_par):

                return "Missing mandatory params",in_p_dict

            else:

                if(i is opt_l_par[0]):

                    in_p_dict[i]=0

                elif(i is opt_l_par[1]):

                    in_p_dict[i]=100

                elif(i is opt_l_par[2]):

                    in_p_dict[i]=None



    

    return False, in_p_dict      
def db_query(mycursor,str_query):

    mycursor.execute(str_query)

    query_res=mycursor.fetchall()

    if (len(query_res)==0):

        return True,query_res

    else:

        return False,query_res
def conn_to_db(in_ps_dict):

    mydb = mysql.connector.connect(

    host="localhost",

    user=username,

    passwd=password,

    database=database_name

    )

    mycursor = mydb.cursor()

    

    #This query acquires Left and Right node information basing on idNode

    str_query="SELECT iLeft,iRight FROM node_tree  WHERE idNode = " + "\'"+str(in_ps_dict['node_id'])+"\'"

   

    error_q,search_results=db_query(mycursor,str_query)

    

    if(error_q is False):

        i_n_left=search_results[0][0]

        i_n_right=search_results[0][1]

        

        if(in_ps_dict['search_keyword'] is not None):    

            #This query restricts the results to "all children nodes under  node_id 

            #whose  nodeName is in the given  language"

            str_query="SELECT t.idNode,iLeft,iRight,nodeName,language FROM node_tree as t INNER JOIN node_tree_names as tn ON t.idNode=tn.idNode WHERE iLeft > "+"\'"+str(i_n_left)+"\'"+" AND iRight < "+"\'"+str(i_n_right)+"\'"+" AND language = "+"\'"+str(in_ps_dict['language'])+"\'"

            error_q,search_results=db_query(mycursor,str_query)

            #This function filters only results which contains  search_keyword (case insensitive)

            search_results=search_keyw(in_ps_dict['search_keyword'],search_results)

            if(error_q is True):

                return "No children node found",None

        else:  

            #This query restricts the results to  node_id 

            #whose  nodeName is in the given  language" -- No keyword input parameter

            str_query="SELECT t.idNode,iLeft,iRight,nodeName,language FROM node_tree as t INNER JOIN node_tree_names as tn ON t.idNode=tn.idNode WHERE t.idNode = "+"\'"+str(in_ps_dict['node_id'])+"\'"+" AND language = "+"\'"+str(in_ps_dict['language'])+"\'"

            error_q,search_results=db_query(mycursor,str_query)

            if(error_q is True):

                return "Missing mandatory params",None

        

         

        mydb.close()

       

        

        return False,search_results

        

        

    else:

        return "Invalid node id",None
def search_keyw(s_key,s_res):

    filt_res=list()

   

    for i in s_res:

        if (s_key in i[3].lower()):

            filt_res.append(i)



    

    return tuple(filt_res)
def pag_func(p_size,p_num,mylist):

    if(p_size!=0):

        if(p_size>len(mylist)):

            p_size=len(mylist)



        p_num_max=round(len(mylist)/p_size)-1

        if(p_num>(p_num_max)):

           p_num=p_num_max

        print(p_size)    

        print(p_num)

        l=[mylist[i:i+p_size] for i in range(0, len(mylist), p_size)]

        print (l)

    

        return l[p_num]

    else:

        return []    
# It creates a Flask app 

app = Flask(__name__) 

  

@app.errorhandler(404) 

def page_not_found(e):

    p_not_f="Please provide information to route localhost  e.g: http://127.0.0.1:5000/?node_id=5&language=italian&search_keyword=r&page_num=2&page_size=3"

    return jsonify({'Error':p_not_f})



@app.route('/') 

def home(): 

    

    if(request.method == 'GET'): 

        in_p_dict=input_acq()

        

        error_p_in,in_ps_dict=set_input(in_p_dict)

        

        #if not found any error in input parameters

        if(error_p_in is False):

            

            res_dict=list()

            

            error_q,search_results=conn_to_db(in_ps_dict)

            #if query gives asome results

            if(error_q is False):

                for x in search_results:

                    #create a dict basing on results collected from query collecting:  unique ID of the child node,

                    # node name translated in the requested language and  number of child nodes of this node

                    res_dict.append({"node_id": x[0],"name": x[3],"children_count": round((x[2]-x[1]-1)/2)})

                    

                 

                #return data collected in a json format

                return jsonify({'nodes':pag_func(int(in_ps_dict["page_size"]),int(in_ps_dict["page_num"]),res_dict) }) 

              

            else:

                #error handling

                return jsonify({'Error':error_q}) 

                

        else:

            #error handling

            return jsonify({'Error':error_p_in}) 
if __name__ == '__main__': 

  

    app.run(debug = True) 