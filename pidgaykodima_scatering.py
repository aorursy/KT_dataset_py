!pip install pywigxjpf argparse # Wigner simbols theorem

! apt install -y gfortran

#!git clone https://gitlab.com/AmosEgel/smuthi.git

!git clone https://gitlab.com/kostyfisik/smuthi.git

# !cd smuthi && git checkout ipython-terminal

!mkdir smuthi_nfmds_bin # 4 cylinders

!mkdir smuthi/smuthi_nfmds_bin # 4 cylinders

!cd smuthi && python setup.py install

!cd smuthi %% git branch

import sys

# import smuthi

default_stdout = sys.stdout
sys.stdout = default_stdout

#*****************************************************************************#

# This is a simple example script for Smuthi v0.8.6.                          #

# It evaluates the differential scattering cross section of a large number of #

# glass spheres on a glass substrate, excited by a plane wave under normal    #

# incidence.                                                                  #

#*****************************************************************************#



import time

import numpy as np

import smuthi.simulation

import smuthi.initial_field

import smuthi.layers

import smuthi.particles

import smuthi.scattered_field

import smuthi.graphical_output

import smuthi.coordinates as coord

import smuthi.cuda_sources as cu

import matplotlib.pyplot as plt

import sys

default_stdout = sys.stdout
import os

output_directory = 'check'

if not os.path.exists(output_directory):

    os.makedirs(output_directory)

file_ext = ".jpg" 
#%%

def combined_plot(data, polarization,incidence_angle_grad):

    sys.stdout = old_stdout # Restore output after muting it in simulation

    Q_sca = []

    Q_ext = []



 # save as txt data

    pol_Name = "TE"

    if polarization == TM: pol_Name = "TM"     

    np.savetxt(output_directory+"/data_"+pol_Name+".txt",data,header="WL sphere_R Q_sca Q_ext")



    

    WLs = np.unique(data[:,0])#  у нас список параметров, в каждом из которых есть одина

                              #  ковый длны волн и толщиныподслоев. из них вытаскиваем то,

                              #что будет основой для нашей сетки

    Ds = np.unique(data[:,1]) # в нашем случае это толщины подслоя,





    Qsca = data[:,2].reshape(len(WLs),len(Ds)).T

    Qext = data[:,3].reshape(len(WLs),len(Ds)).T      

    Qlist = [Qsca,Qext]

    Qname = ["Qsca",'Qext']



    vscale = [1,1]

    min_tick = [0.0*10**5, 0.0*10**5]

    

    for q, name, vscale,min_tick in list(zip(Qlist,Qname,vscale,min_tick)):



        fig, ax = plt.subplots(figsize=(16,9))



        cax = ax.imshow(q

                        , interpolation='nearest'                      

                        , cmap='jet',

                        origin='lower'                      

                            ,extent=(min(WLs), max(WLs), min(Ds), max(Ds))

                        )

        ax.set_aspect('auto')

        

        max_tick = np.amax(q[~np.isnan(q)])*vscale 

        scale_ticks = np.linspace(min_tick, max_tick, 10) # количество засечек на колорбаре

        cbar = fig.colorbar(cax, ticks=[a for a in scale_ticks], ax=ax, fraction=0.046, pad=0.04)

        

        plt.title(name+"_"+pol_Name+"\n", fontsize=18)

        plt.xlabel(r'$\lambda, nm$', fontsize=18)

        plt.ylabel(r'$h_{buffer},nm$', fontsize=18)

        plt.savefig(output_directory+"/plot_"+name+"_"+pol_Name+file_ext)

                    

        plt.close()



        

        

    # 2D plot

    if (True):

        max_tick = [25,8]

        for q, name, max_tick in list(zip(Qlist,Qname,max_tick)):

            fig, ax = plt.subplots(figsize=(6,6))

            scale = 10**6

            for i in range(len(Ds)):

                ax.plot(WLs,q[i]/scale,label=str(Ds[i]))

            # cax = ax.imshow(q

            #                     , extent=(min(WLs), max(WLs), min(Ds), max(Ds))

            legend = ax.legend()

            #ax.set_ylim(0,max_tick)

            plt.title(name+", "+pol_Name+", NA ="+str(collect_NA)+", in_ang ="+str(incidence_angle*180/np.pi))

            plt.xlabel(r'$\lambda$, nm')

            plt.ylabel(name+r'$, \mu m^2$')

            plt.savefig(output_directory+"/"+name+", "+pol_Name+", NA ="+str(collect_NA)+", in_ang ="+str(incidence_angle*180/np.pi)+file_ext)

            # plt.savefig(output_directory+"/plot1D_"+name+"_"+pol_Name+suffix+file_ext)

            plt.close()
def GetTopTSCS(WL, index_NP, index_buffer, h_buffer, index_substrate, sphere_R, polarization, incidence_agle, collect_NA, BFP):

    #print(WL, index_NP, index_buffer, h_buffer, index_substrate, sphere_R, polarization)

    spacer = 2 #nm

    # Initialize a plane wave object the initial field

    

    l_max = 1

    neff_max = 5

    

    #global polar_ang

    plane_wave = smuthi.initial_field.PlaneWave(

        vacuum_wavelength=WL,

        polar_angle= incidence_agle,  # 25 grad to the surface

        #polar_angle= polar_ang,  

        azimuthal_angle=0,

        polarization=polarization)           # 0 stands for TE, 1 stands for TM

    pol_name = "TE"

    if polarization == 1: pol_name = "TM"



    # Initialize the layer system object

    #The coordinate system is such that the interface between the

    # first two layers defines the plane z=0.



    layer_system = smuthi.layers.LayerSystem(

        thicknesses=[0,  h_buffer, 0],         #  ambient, substrate

        refractive_indices=[1, index_buffer, index_substrate])   # like air,glass





    # Define the scattering particles

    particle_grid = []



    

    sphere = smuthi.particles.Sphere(position=[0, 0, -sphere_R-spacer],

                                     refractive_index=index_NP,

                                     radius=sphere_R,

                                     l_max=l_max)    # choose l_max with regard to particle size and material

                                                 # higher means more accurate but slower

    particle_grid.append(sphere)



    # Define contour for Sommerfeld integral

    smuthi.coordinates.set_default_k_parallel(vacuum_wavelength=plane_wave.vacuum_wavelength,

                                              neff_resolution=5e-3,       # smaller value means more accurate but slower

                                              neff_max=neff_max)                 # should be larger than the highest refractive

                                                                          # index of the layer system



    # Initialize and run simulation

    simulation = smuthi.simulation.Simulation(layer_system=layer_system,

                                              particle_list=particle_grid,

                                              initial_field=plane_wave,

                                              solver_type='LU',

                                              # solver_type='gmres',

                                              solver_tolerance=1e-5,

                                              store_coupling_matrix=True,

                                              coupling_matrix_lookup_resolution=None,

                                              # store_coupling_matrix=False,

                                              # coupling_matrix_lookup_resolution=5,

                                              coupling_matrix_interpolator_kind='cubic',

                                              log_to_file=False

                                              ,log_to_terminal = False

                                                  )

    simulation.run()

    



    p_angles = np.linspace(0, np.pi, integral_samples, dtype=float)

    collect_angle = np.arcsin(collect_NA/n_air)

    #print("Collecting angle for output objective =", collect_angle)

    

    p_angles = np.linspace(np.pi/2, np.pi, integral_samples, dtype=float)

    a_angles = np.linspace(0, 2.0*np.pi, integral_samples, dtype=float)

    #print(angles)

    scattering_cross_section = smuthi.scattered_field.scattering_cross_section(

        initial_field=plane_wave,

        particle_list=particle_grid,

        layer_system=layer_system

        ,polar_angles=p_angles

        ,azimuthal_angles=a_angles

        )



    extinction_cross_section = smuthi.scattered_field.extinction_cross_section(

        initial_field=plane_wave, particle_list=particle_grid,

        layer_system=layer_system)



    Q_sca = (#scattering_cross_section.top().integral()[0]

              #+scattering_cross_section.top().integral()[1]

                +scattering_cross_section.bottom().integral()[0]

                 + scattering_cross_section.bottom().integral()[1]

    ).real/(np.pi*sphere_R**2)

    sys.stdout = old_stdout

    

    if BFP:

        smuthi.graphical_output.show_far_field(scattering_cross_section, 

                                        save_plots=True,

                                        show_plots=False,

                                        save_data=True,

                                        tag = str(int(WL)) ,

                                        outputdir= output_directory+'/Directivity_' + pol_name+'/')#,

                                        #Norm=np.pi*sphere_R**2)#+str(WL))





    Q_ext = (extinction_cross_section['top'] + extinction_cross_section['bottom']).real/(np.pi*sphere_R**2)



    plot_size = WL/2/2 

    

    

    sys.stdout= old_stdout

    return Q_sca, Q_ext
#%%

integral_samples = 180 

incidence_angle_grad = 0

from_WL = 100

to_WL = 800

total_points = 31



WLs = np.linspace(from_WL, to_WL, total_points)





n_media = 1.0

n_air = 1.0



from_R = 85

to_R = 85

total_R_points =1

R_list = np.linspace(from_R,to_R,total_R_points)

#sphere_R = 150





from_h_buffer = 280

to_h_buffer = 280

total_h_points = 31

h_list = np.linspace(from_h_buffer, to_h_buffer, total_h_points)



incidence_angle= incidence_angle_grad*(np.pi/180)





 

source_NA = 1 # Numerical aperture for the input objective, not

                 # used for plane wave exitation

collect_NA = 1 # Collect apperture

collect_angle = np.arcsin(collect_NA/n_media)



# Au_thickness = 40 #nm

Au_thickness = 0

core_r = from_R



TE = 0

TM = 1 

BFP = False



#%%



Q_sca = []

Q_ext = []

old_stdout = default_stdout

preparation_time = 0

solution_time = 0

cu.enable_gpu(True)

if not cu.use_gpu:

    default_stdout.write("Failed to load pycuda")


start = time.time()

for sphere_R in R_list:

   for h_buffer in h_list:

       for i in range(len(WLs)): 



        index_NP = 4

        index_buf = 1

        index_subst = 1

        sys.stdout = old_stdout

        progress = i*100.0/len(WLs)

        sys.stdout = old_stdout

        print(str(int(progress))+'%')

        #PrintProgress(progress)



        valTE, valTE_ext = GetTopTSCS(WLs[i], index_NP, index_buf, h_buffer, index_subst, sphere_R, TE, incidence_angle, collect_NA, BFP)

        valTM, valTM_ext = GetTopTSCS(WLs[i], index_NP, index_buf, h_buffer, index_subst, sphere_R, TM, incidence_angle, collect_NA, BFP)

        Q_sca.append(np.hstack((valTE, valTM)))

        Q_ext.append(np.hstack((valTE_ext, valTM_ext)))

        

end = time.time()

solution_time = end - start

solution_time    
Q_sca1 = np.array(Q_sca)



plt.subplots(figsize=(12,9))

plt.plot( WLs, Q_sca1, linewidth = 3)

plt.legend(["$0$,0"],prop={'size': 34})



plt.xlabel(r'$\lambda, nm$',fontsize = 34, fontname = "Times New Roman")

plt.ylabel(r'$C_{sca}$',fontsize = 34, fontname = "Times New Roman")

plt.xticks(fontsize = 34, fontname = "Times New Roman")

plt.yticks(fontsize = 34, fontname = "Times New Roman")



plt.savefig(output_directory+"/Q_sca_spectra_on Si.png")
