import kagglegym
kagglegym.Space
kagglegym.__dict__
env = kagglegym.make()
env.__dict__
env._gs
import inspect
inspect.getmembers(env.reset)
env.__class__
env.reset.__func__.__code__
import dis
dis.dis(env.reset.__func__.__code__)
dis.dis(env.step.__func__.__code__)
kagglegym.GaussianEnv.TARGET_COL_NAME
dis.dis(env._gs.__func__.__code__)
env._code_submission_output
kagglegym.GaussianEnv.__dict__
dis.dis(kagglegym.GaussianEnv.__init__.__code__)
kagglegym.GaussianEnv._get_state_iterator()